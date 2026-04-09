from simulation.base.abstract_avatar import abstract_avatar
from simulation.memory import AvatarMemory
from simulation.llm_client import request_completion

from termcolor import colored, cprint
import os

import re
import numpy as np
import faiss
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.chat_models import ChatOpenAI
from simulation.retriever import AvatarRetriver
import time
import datetime
import torch
from langchain.embeddings import OpenAIEmbeddings
import pandas as pd
import hashlib

import wandb

import simulation.vars as vars

class Avatar(abstract_avatar):
    def __init__(self, args, avatar_id, init_property, init_statistic):
        super().__init__(args, avatar_id)
        dataset_name = str(args.dataset).lower()
        if "book" in dataset_name:
            self.prompt_profile = "book"
            self.item_word = "book"
        elif "beauty" in dataset_name or "amazon" in dataset_name:
            self.prompt_profile = "beauty"
            self.item_word = "beauty product"
        elif "kuairand" in dataset_name:
            self.prompt_profile = "short_video"
            self.item_word = "short video"
        else:
            self.prompt_profile = "legacy_movie"
            self.item_word = "movie"
        self.use_beauty_prompt = self.prompt_profile == "beauty"

        self.parse_init_property(init_property)
        self.parse_init_statistic(init_statistic)
        # Session intent is only used for beauty prompts (prompt_mode=c).
        self.session_profile = {}
        if self.prompt_profile == "beauty":
            self.session_profile = self._infer_beauty_session_profile()
        # Exit after accumulated negative feedback reaches threshold.
        self.negative_feedback_count = 0
        self.log_file = f"storage/{args.dataset}/{args.modeltype}/{args.simulation_name}/running_logs/{avatar_id}.txt"
        if os.path.exists(self.log_file):
            os.remove(self.log_file)
        self.init_memory()
        self._seed_profile_memory()

    @staticmethod
    def _compact_timeline_text(
        timeline: str,
        *,
        max_events: int = 8,
        max_each_chars: int = 220,
        max_total_chars: int = 1500,
    ) -> str:
        """
        Persona timelines can be long and are repeated on every page.
        Keep the most recent events while preserving chronological order.
        """
        text = str(timeline or "").strip()
        if not text:
            return ""

        parts = [p.strip() for p in text.split("||") if p.strip()]
        if len(parts) > max_events:
            parts = parts[-max_events:]
        clipped = []
        for p in parts:
            p = re.sub(r"\s+", " ", p).strip()
            if len(p) > max_each_chars:
                p = p[: max_each_chars - 3].rstrip() + "..."
            clipped.append(p)
        out = " || ".join(clipped).strip()
        if len(out) > max_total_chars:
            out = out[-max_total_chars:]
        return out

    @staticmethod
    def _safe_int_env(name: str, default: int) -> int:
        raw = os.getenv(name, str(default))
        try:
            value = int(raw)
        except Exception:
            value = default
        return max(1, value)

    @staticmethod
    def _extract_id_items(recommended_items_str):
        items = []
        text = str(recommended_items_str or "")
        block_re = re.compile(
            r"<-\s*ID:\s*(\d+)\s*->\s*<-\s*Title:\s*(.*?)\s*->",
            re.IGNORECASE | re.DOTALL,
        )
        for match in block_re.finditer(text):
            try:
                item_id = int(match.group(1))
            except Exception:
                continue
            title = re.sub(r"\s+", " ", str(match.group(2) or "")).strip()
            items.append({"id": item_id, "title": title})
        return items

    @staticmethod
    def _extract_legacy_titles(recommended_items_str):
        titles = []
        for line in str(recommended_items_str or "").splitlines():
            match = re.search(r"<-\s*(.*?)\s*->", line)
            if not match:
                continue
            title = re.sub(r"\s+", " ", str(match.group(1) or "")).strip()
            if title:
                titles.append(title)
        return titles

    def _fallback_recommended_items_response(self, recommended_items_str):
        if self.prompt_profile in {"beauty", "short_video", "book"}:
            items = self._extract_id_items(recommended_items_str)
            lines = [
                f"MOVIE: {item['id']}; ALIGN: no; REASON: api timeout, skip this item."
                for item in items
            ]
            lines.append("NUM: 0; WATCH: None; REASON: api timeout, stop clicking;")
            return "\n".join(lines)

        titles = self._extract_legacy_titles(recommended_items_str)
        lines = [
            f"MOVIE: {title}; WATCH: no; REASON: api timeout; RATING: 1; FEELING: timeout fallback."
            for title in titles
        ]
        return "\n".join(lines)

    @staticmethod
    def _fallback_scan_response():
        return "SHORTLIST: None;\nINFO_NEEDED: api timeout;"

    @staticmethod
    def _fallback_next_decision_response():
        return "NEGATIVE: api timeout.\n[EXIT]; Reason: api timeout."

    @staticmethod
    def _fallback_interview_response():
        return "RATING: 3; REASON: session evaluation unavailable because the api request timed out."

    def _infer_beauty_session_profile(self):
        """
        Create a deterministic, mission-driven session intent so the avatar
        behaves like a real shopper (conservative clicks). This is used by
        beauty_prompt_mode=c.
        """

        # Extract product-type preferences from the persona taste field.
        type_prefs = []

        def canonical_type(text: str):
            """
            Map noisy persona/category strings into a small set of product types
            that are likely to be supported by the item Tags/Summary fields.
            """
            t = str(text or "").lower()
            if any(k in t for k in ["cleanser", "cleanse", "cleansing", "face wash"]):
                return "cleanser"
            if any(k in t for k in ["hair", "shampoo", "conditioner", "scalp"]):
                return "hair care"
            if "mask" in t:
                return "mask"
            if any(k in t for k in ["makeup", "mascara", "eyeliner", "foundation", "bb cream", "concealer", "lip", "eyelash"]):
                return "makeup"
            if "nail" in t:
                return "nail care"
            if any(k in t for k in ["fragrance", "perfume", "scent"]):
                return "fragrance"
            if any(k in t for k in ["sunscreen", "spf", "sun care"]):
                return "sun care"
            if any(k in t for k in ["body wash", "bath", "scrub", "body care"]):
                return "body care"
            if any(k in t for k in ["serum", "moistur", "cream", "toner", "oil", "lotion", "skincare", "skin care"]):
                return "skincare"
            return None

        for raw in self.taste or []:
            m = re.search(r"prefer\s+(.+?)\s+products", str(raw), flags=re.IGNORECASE)
            if not m:
                continue
            cat = m.group(1).strip().strip(".")
            if not cat:
                continue
            # Skip brand-like preferences (e.g., "Brand:XYZ") and other colon tags.
            if ":" in cat:
                continue
            canon = canonical_type(cat)
            if canon:
                type_prefs.append(canon)

        if not type_prefs:
            # Fallback: parse the high_rating tendency line.
            m = re.search(r"tends to rate\s+(.+?)\s+products\s+highly", str(self.high_rating), flags=re.IGNORECASE)
            if m:
                raw = m.group(1)
                for part in str(raw).split(","):
                    cat = part.strip().strip(".")
                    if not cat or ":" in cat:
                        continue
                    canon = canonical_type(cat)
                    if canon:
                        type_prefs.append(canon)

        if not type_prefs:
            type_prefs = ["skincare"]
        else:
            type_prefs = list(dict.fromkeys(type_prefs))

        focus = type_prefs[int(self.avatar_id) % len(type_prefs)]

        # Mission type: low-activity users are "restock", high-activity are "explore".
        if int(getattr(self, "activity_group", 2)) <= 2:
            mission = "restock"
        else:
            mission = "explore"

        # Conservative numeric gates. These are intentionally strict to reduce
        # false-positive clicks for weak recommenders.
        activity = int(getattr(self, "activity_group", 2))
        if mission == "restock":
            if activity == 1:
                watch_rating_min, watch_review_min = 4.4, 200
            elif activity == 2:
                watch_rating_min, watch_review_min = 4.2, 120
            else:
                watch_rating_min, watch_review_min = 4.1, 80
        else:  # explore
            if activity == 1:
                watch_rating_min, watch_review_min = 4.4, 180
            elif activity == 2:
                watch_rating_min, watch_review_min = 4.2, 100
            else:
                watch_rating_min, watch_review_min = 4.1, 50

        align_rating_min = max(3.8, float(watch_rating_min) - 0.2)
        align_review_min = max(20, int(watch_review_min) // 2)

        items_per_page = int(getattr(self.args, "items_per_page", 1) or 1)
        max_watch = 1 if items_per_page <= 3 else 2

        # Calibration target used only in prompting (not enforced in code).
        target_watch_rate = 0.08 if mission == "restock" else 0.12
        if activity == 1:
            target_watch_rate *= 0.7
        elif activity == 3:
            target_watch_rate *= 1.2
        target_watch_rate = float(min(max(target_watch_rate, 0.03), 0.25))

        return {
            "mission": mission,
            "focus": focus,
            "align_rating_min": float(align_rating_min),
            "align_review_min": int(align_review_min),
            "watch_rating_min": float(watch_rating_min),
            "watch_review_min": int(watch_review_min),
            "max_watch": int(max_watch),
            "target_watch_rate": float(target_watch_rate),
        }

    def _seed_profile_memory(self):
        # Keep legacy movie behavior unchanged; seed extra context for structured prompts.
        if self.prompt_profile == "legacy_movie":
            return
        taste = "; ".join(self.taste).strip()
        timeline = self._compact_timeline_text(self.timeline).strip()
        high = self.high_rating.strip()
        low = self.low_rating.strip()
        parts = ["My long-term preferences:", taste or "unknown"]
        if high:
            parts.append(f"High-rating tendency: {high}")
        if low:
            parts.append(f"Low-rating tendency: {low}")
        if timeline:
            parts.append(f"Chronological history: {timeline}")
        if self.session_profile:
            parts.append(
                "Session intent: "
                + f"{self.session_profile.get('mission','').upper()} "
                + f"{self.session_profile.get('focus','')}"
            )
        # This is a pure memory seed, no LLM call.
        try:
            self.memory.add_memory(" | ".join(parts), now=datetime.datetime.now())
        except Exception:
            # Fail open: simulation can still run even if memory write fails.
            pass

    @staticmethod
    def _clean_optional_text(value):
        if value is None:
            return ""
        text = str(value).strip()
        if text.lower() in {"nan", "none", "null"}:
            return ""
        return text

    @staticmethod
    def _dedupe_text_keep_order(values):
        out = []
        seen = set()
        for value in values:
            text = Avatar._clean_optional_text(value).strip().strip(";")
            if not text:
                continue
            if text in seen:
                continue
            seen.add(text)
            out.append(text)
        return out

    def _extract_page_id_title_map(self, recommended_items_str):
        mapping = {}
        if not recommended_items_str:
            return mapping
        pattern = re.compile(
            r"<-\s*ID\s*:\s*(\d+)\s*->\s*<-\s*Title\s*:\s*(.*?)\s*->",
            re.IGNORECASE,
        )
        for item_id, title in pattern.findall(str(recommended_items_str)):
            title_clean = self._clean_optional_text(title)
            if title_clean:
                mapping[str(item_id)] = title_clean
        return mapping

    def _normalize_page_token(self, raw_token, id_title_map):
        token = self._clean_optional_text(raw_token).strip().strip(";")
        if not token:
            return ""
        if token in id_title_map:
            return id_title_map[token]
        return token

    def parse_init_property(self, init_property):
        if self.prompt_profile == "legacy_movie":
            # Preserve the original MovieLens formatting as much as possible.
            taste_field = self._clean_optional_text(init_property.get("taste", ""))
            self.taste = taste_field.split("| ") if taste_field else []
            self.high_rating = init_property.get("high_rating", "")
            self.low_rating = init_property.get("low_rating", "")
            self.timeline = ""
            return

        taste_raw = self._clean_optional_text(init_property.get("taste", ""))
        self.taste = [part.strip() for part in taste_raw.split("|") if part.strip()]
        if not self.taste:
            self.taste = [f"I care about quality, fit and value when choosing a {self.item_word}."]
        self.high_rating = self._clean_optional_text(init_property.get("high_rating", ""))
        self.low_rating = self._clean_optional_text(init_property.get("low_rating", ""))
        self.timeline = self._clean_optional_text(init_property.get("timeline", ""))


    def parse_init_statistic(self, init_statistic):
        """
        Parse the init statistic of the avatar
        """
        if self.prompt_profile == "beauty":
            activity_dict = {
                1: "An Ultra-Minimalist Shopper who rarely tries new products. You are impatient with weak matches and tend to leave quickly after a few disappointing pages.",
                2: "A Routine-Oriented Shopper who mostly sticks to trusted product types. You will consider a few strong matches and leave if recommendations feel irrelevant.",
                3: "An Active Experimenter who enjoys browsing and trying products. You are tolerant of imperfect pages and keep exploring if there is any potential fit.",
            }
            conformity_dict = {
                1: "A Rating-Conforming Shopper who is heavily influenced by average ratings and review consensus, and tends to follow the crowd.",
                2: "A Balanced Shopper who considers both average ratings and your own needs; you sometimes disagree with the crowd.",
                3: "An Independent Critic who mostly ignores average ratings and decides based on personal fit and preferences.",
            }
            diversity_dict = {
                1: "A Category Specialist who sticks to a narrow set of product types/brands and avoids unfamiliar options.",
                2: "A Selective Explorer who occasionally tries adjacent product types but usually stays within your comfort zone.",
                3: "A Curious Explorer who often tries new brands/product types and is open to novelty.",
            }
        elif self.prompt_profile == "short_video":
            activity_dict = {
                1: "A Low-Activity Viewer who quickly skips weak recommendations and usually leaves after a few poor pages.",
                2: "A Moderate Viewer who watches selectively; you continue browsing only when content relevance stays acceptable.",
                3: "A Highly Active Viewer who can browse many pages and explore broader short-video topics.",
            }
            conformity_dict = {
                1: "A Trend-Following Viewer who is strongly influenced by popularity and public feedback.",
                2: "A Balanced Viewer who combines personal taste with popularity signals.",
                3: "An Independent Viewer who prioritizes personal taste over popularity.",
            }
            diversity_dict = {
                1: "A Focused Viewer who prefers a narrow range of short-video topics.",
                2: "A Selective Explorer who occasionally tries nearby topics.",
                3: "A Broad Explorer who actively watches diverse short-video topics.",
            }
        elif self.prompt_profile == "book":
            activity_dict = {
                1: "A Goal-Driven Reader who leaves quickly when recommended books miss your interests or feel not worth the time.",
                2: "A Selective Reader who will browse a few pages when the books stay relevant, but you do not tolerate many weak matches.",
                3: "An Active Book Browser who enjoys exploring shelves and can stay longer when there is clear thematic promise.",
            }
            conformity_dict = {
                1: "A Review-Driven Reader who is strongly influenced by average ratings and reader consensus.",
                2: "A Balanced Reader who considers both public ratings and your own reading taste.",
                3: "An Independent Reader who mainly follows personal interests over crowd opinion.",
            }
            diversity_dict = {
                1: "A Genre Specialist who usually stays within a narrow set of topics, genres, or trusted authors.",
                2: "A Selective Explorer who sometimes tries adjacent genres or unfamiliar authors.",
                3: "A Broad Reader who is open to diverse genres, subjects, and reading styles.",
            }
        else:
            # Legacy MovieLens wording (keep unchanged).
            activity_dict = {   1:"An Incredibly Elusive Occasional Viewer, so seldom attracted by movie recommendations that it's almost a legendary event when you do watch a movie. Your movie-watching habits are extraordinarily infrequent. And you will exit the recommender system immediately even if you just feel little unsatisfied.",
                                2:"An Occasional Viewer, seldom attracted by movie recommendations. Only curious about watching movies that strictly align the taste. The movie-watching habits are not very infrequent. And you tend to exit the recommender system if you have a few unsatisfied memories.",
                                3:"A Movie Enthusiast with an insatiable appetite for films, willing to watch nearly every movie recommended to you. Movies are a central part of your life, and movie recommendations are integral to your existence. You are tolerant of recommender system, which means you are not easy to exit recommender system even if you have some unsatisfied memory."}
            conformity_dict = { 1:"A Dedicated Follower who gives ratings heavily relies on movie historical ratings, rarely expressing independent opinions. Usually give ratings that are same as historical ratings. ",
                                2:"A Balanced Evaluator who considers both historical ratings and personal preferences when giving ratings to movies. Sometimes give ratings that are different from historical rating.",
                                3:"A Maverick Critic who completely ignores historical ratings and evaluates movies solely based on own taste. Usually give ratings that are a lot different from historical ratings."}
            diversity_dict = {  1:"An Exceedingly Discerning Selective Viewer who watches movies with a level of selectivity that borders on exclusivity. The movie choices are meticulously curated to match personal taste, leaving no room for even a hint of variety.",
                                2:"A Niche Explorer who occasionally explores different genres and mostly sticks to preferred movie types.",  
                                3:"A Cinematic Trailblazer, a relentless seeker of the unique and the obscure in the world of movies. The movie choices are so diverse and avant-garde that they defy categorization."}
        
        self.conformity_group = init_statistic["conformity"]
        self.activity_group = init_statistic["activity"]
        self.diversity_group = init_statistic["diversity"]
        self.conformity_dsc = conformity_dict[self.conformity_group]
        self.activity_dsc = activity_dict[self.activity_group]
        self.diversity_dsc = diversity_dict[self.diversity_group]
        if self.prompt_profile in {"beauty", "short_video", "book"}:
            self.exit_threshold = {1: 1, 2: 2, 3: 3}.get(self.activity_group, 2)
        else:
            self.exit_threshold = 2

    def init_memory(self):
        """
        Initialize the memory of the avatar
        """
        t1 = time.time()
        def score_normalizer(val: float) -> float:
            return 1 - 1 / (1 + np.exp(val))

        class LocalHashEmbeddings:
            def __init__(self, size=256):
                self.size = size

            def _embed(self, text):
                seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) % (2**32)
                rng = np.random.default_rng(seed)
                vec = rng.standard_normal(self.size).astype(np.float32)
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                return vec.tolist()

            def embed_query(self, text):
                return self._embed(text)

            def embed_documents(self, texts):
                return [self._embed(t) for t in texts]

        use_openai_embeddings = os.getenv("USE_OPENAI_EMBEDDINGS", "0") == "1"
        if use_openai_embeddings:
            embed_model_name = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
            embeddings_model = OpenAIEmbeddings(
                request_timeout=20,
                model=embed_model_name,
            )
            embedding_size = len(embeddings_model.embed_query("dimension_probe"))
        else:
            embeddings_model = LocalHashEmbeddings(size=256)
            embedding_size = embeddings_model.size

        index = faiss.IndexFlatL2(embedding_size)
        vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {}, relevance_score_fn=score_normalizer)

        LLM = ChatOpenAI(max_tokens=1000, temperature=0.3, request_timeout = 30)
        avatar_retriever = AvatarRetriver(vectorstore=vectorstore, k=5)
        self.memory = AvatarMemory(
            memory_retriever=avatar_retriever,
            llm=LLM,
            reflection_threshold=3,
            use_wandb=self.use_wandb,
            llm_model=self.args.llm_model,
            llm_api_style=self.args.llm_api_style,
        )
        t2 = time.time()

        
        cprint(f"Avatar {self.avatar_id} is initialized with memory", color='green', attrs=['bold'])
        cprint(f"Time cost: {t2-t1}s", color='green', attrs=['bold'])



    def _reaction(self, messages=None, timeout=30, fallback_response=None, context_label="reaction"):
        """
        Summarize the feelings of the avatar for recommended item list.
        """ 
        response = ''
        except_waiting_time = 1
        max_waiting_time = 16
        current_sleep_time = 0.5
        max_attempts = self._safe_int_env("AVATAR_LLM_MAX_RETRIES", 4)
        attempts = 0
        last_error = ""
        while response == '' and attempts < max_attempts:
            try:
                start_time = time.time()
                time_local = time.localtime(start_time)
                l_start = time.strftime("%Y-%m-%d %H:%M:%S",time_local)

                if(self.use_wandb): # whether to use wandb
                    if((start_time - vars.global_start_time)//vars.global_interval > vars.global_steps):
                        print("\nStart Identifier", start_time, vars.global_start_time, (start_time - vars.global_start_time), vars.global_steps)
                        if(vars.lock.acquire(False)):
                            print("\nStart Identifier", start_time, vars.global_start_time, (start_time - vars.global_start_time), vars.global_steps)
                            vars.global_steps += 1
                            wandb.log(
                                data = {"Real-time Traffic": vars.global_k_tokens - vars.global_last_tokens_record,
                                        "Total Traffic": vars.global_k_tokens,
                                        "Finished Users": vars.global_finished_users,
                                        "Finished Pages": vars.global_finished_pages,
                                        "Error Cast": vars.global_error_cast/1000,
                                },
                                step = vars.global_steps
                            )
                            vars.global_last_tokens_record = vars.global_k_tokens
                            vars.lock.release()
                            print("\nEnd Identifier", time.time(), vars.global_start_time, (time.time() - vars.global_start_time), vars.global_steps)
                            
                completion = request_completion(
                    messages=messages,
                    model=self.args.llm_model,
                    temperature=float(getattr(self.args, "llm_temperature", 0.0)),
                    timeout=timeout,
                    max_tokens=1000,
                    api_style=self.args.llm_api_style,
                )

                l_end = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))
                k_tokens = completion["k_tokens"]
                print(f"User {self.avatar_id} used {k_tokens} tokens from {l_start} to {l_end}")
                self.memory.user_k_tokens += k_tokens
                vars.global_k_tokens += k_tokens
                response = str(completion["content"] or "").strip()
                if response == "":
                    raise ValueError("Empty response from LLM")
            except Exception as e:
                attempts += 1
                last_error = f"{type(e).__name__}: {e}"
                vars.global_error_cast += 1
                self.write_log(
                    f"[LLM][{context_label}] attempt {attempts}/{max_attempts} failed: {last_error}",
                    color="red",
                )
                if attempts >= max_attempts:
                    break
                time.sleep(current_sleep_time)
                if except_waiting_time < max_waiting_time:
                    except_waiting_time *= 2
                current_sleep_time = np.random.randint(0, except_waiting_time-1)

        if response != "":
            return response

        if fallback_response is not None:
            self.write_log(
                f"[LLM][{context_label}] fallback used after {max_attempts} failed attempts. Last error: {last_error}",
                color="red",
            )
            return fallback_response

        raise RuntimeError(
            f"LLM request failed after {max_attempts} attempts in {context_label}. Last error: {last_error}"
        )

    @staticmethod
    def _is_negative_decision(decision_text):
        text = (decision_text or "").lower()
        first_nonempty = ""
        for raw_line in str(decision_text or "").splitlines():
            line = raw_line.strip()
            if line:
                first_nonempty = line.lower()
                break
        if first_nonempty.startswith("positive:"):
            return False
        if first_nonempty.startswith("negative:"):
            return True
        negative_markers = [
            "negative:",
            "unsatisfied",
            "dissatisfied",
            "not satisfied",
            "poor match",
        ]
        return any(marker in text for marker in negative_markers)
    
    
    def make_next_decision(self, remember=False, current_page=None):
        observation = "Do you satisfy with current recommendation system and what's your interaction history?"
        relevant_memories = self.memory.fetch_memories(observation)
        formated_relevant_memories = self.memory.format_memories_detail(relevant_memories)
        if self.prompt_profile == "beauty":
            sys_prompt = (f"You excel at role-playing. Picture yourself as a user exploring a {self.item_word} recommendation system. You have the following social traits: "
                        +f"\nYour activity trait is described as: {self.activity_dsc}"
                        +f"\nNow you are in Page {current_page}."
                    +f"\nCurrent accumulated negative feedback count: {self.negative_feedback_count}. Exit threshold: {self.exit_threshold}."
                        +f"\nRelevant context from your memory:"
                        +f"\n{formated_relevant_memories}"
                        )
        elif self.prompt_profile == "book":
            sys_prompt = (f"You excel at role-playing. Picture yourself as a user exploring a {self.item_word} recommendation system. You have the following social traits: "
                        +f"\nYour activity trait is described as: {self.activity_dsc}"
                        +f"\nNow you are on page {current_page}."
                        +f"\nCurrent accumulated negative feedback count: {self.negative_feedback_count}. Exit threshold: {self.exit_threshold}."
                        +f"\nRelevant context from your memory:"
                        +f"\n{formated_relevant_memories}"
                        )
        elif self.prompt_profile == "short_video":
            sys_prompt = (f"You excel at role-playing. Picture yourself as a user exploring a short-video recommendation system. You have the following social traits: "
                        +f"\nYour activity trait is described as: {self.activity_dsc}"
                        +f"\nNow you are on page {current_page}."
                        +f"\nCurrent accumulated negative feedback count: {self.negative_feedback_count}. Exit threshold: {self.exit_threshold}."
                        +f"\nRelevant context from your memory:"
                        +f"\n{formated_relevant_memories}"
                        )
        else:
            sys_prompt = ("You excel at role-playing. Picture yourself as a user exploring a movie recommendation system. You have the following social traits: " \
                        +f"\nYour activity trait is described as: {self.activity_dsc}"
                        +f"\nNow you are in Page {current_page}."
                        +f"\nCurrent accumulated negative feedback count: {self.negative_feedback_count}. Exit threshold: {self.exit_threshold}."
                        +f"\nRelevant context from your memory:"
                        +f"\n{formated_relevant_memories}"
                        )
        prompt = ("Firstly, generate an overall feeling based on your memory, in accordance with your activity trait and your satisfaction on recommender system."
                +"\nIf your overall feeling is positive, write: POSITIVE: [reason]"
                +"\nIf it's negative, write: NEGATIVE: [reason]"
                +"\nNow decide whether to continue browsing or exit."
                +f"\nRule: only exit when accumulated negative feedback reaches {self.exit_threshold}."
                +"\nIf this page is NEGATIVE, it adds 1 to accumulated negative feedback."
                +"\nTo leave, write: [EXIT]; Reason: [brief reason]"
                +"\nTo continue browsing, write: [NEXT]; Reason: [brief reason]"
            )
        messages = [{"role": "system",
                    "content": sys_prompt},
                    {"role": "user",
                    "content": prompt}]
        
        self.write_log("\n" + sys_prompt, color="blue")
        self.write_log("\n" + prompt, color="blue")
        response = self._reaction(
            messages,
            fallback_response=self._fallback_next_decision_response(),
            context_label="make_next_decision",
        )
        self.write_log("\n" + response, color="white")

        return response
    
    def response_to_question(self, question, remember=False):
        relevant_memories = self.memory.memory_retriever.memory_stream
        formated_relevant_memories = self.memory.format_memories_detail(relevant_memories)
        if self.prompt_profile == "beauty":
            sys_prompt = (f"You excel at role-playing. Picture yourself as user {self.avatar_id} who has just finished exploring a {self.item_word} recommendation system. You have the following social traits:"
                    +f"\nYour activity trait is described as: {self.activity_dsc}"
                    +f"\nYour conformity trait is described as: {self.conformity_dsc}"
                    +f"\nYour diversity trait is described as: {self.diversity_dsc}"
                    +f"\nBeyond that, your {self.item_word} tastes are: {'; '.join(self.taste).replace('I ','')}. "
                    +f"\nThe activity characteristic pertains to the frequency of your {self.item_word} browsing and consumption habits. The conformity characteristic measures the degree to which your ratings are influenced by historical ratings. The diversity characteristic gauges your likelihood of trying {self.item_word}s that may not align with your usual taste."
                    )
        elif self.prompt_profile == "book":
            sys_prompt = (f"You excel at role-playing. Picture yourself as user {self.avatar_id} who has just finished exploring a {self.item_word} recommendation system. You have the following social traits:"
                    +f"\nYour activity trait is described as: {self.activity_dsc}"
                    +f"\nYour conformity trait is described as: {self.conformity_dsc}"
                    +f"\nYour diversity trait is described as: {self.diversity_dsc}"
                    +f"\nBeyond that, your {self.item_word} tastes are: {'; '.join(self.taste).replace('I ','')}. "
                    +"\nThe activity trait reflects how long you keep browsing books. The conformity trait reflects how much ratings and reader consensus affect your judgement. The diversity trait reflects openness to new genres, topics, and authors."
                    )
        elif self.prompt_profile == "short_video":
            sys_prompt = (f"You excel at role-playing. Picture yourself as user {self.avatar_id} who has just finished exploring a short-video recommendation system. You have the following social traits:"
                    +f"\nYour activity trait is described as: {self.activity_dsc}"
                    +f"\nYour conformity trait is described as: {self.conformity_dsc}"
                    +f"\nYour diversity trait is described as: {self.diversity_dsc}"
                    +f"\nBeyond that, your short-video tastes are: {'; '.join(self.taste).replace('I ','')}. "
                    +"\nThe activity trait reflects how long you keep browsing videos. The conformity trait reflects how much popularity cues affect your judgement. The diversity trait reflects openness to new video topics."
                    )
        else:
            sys_prompt = (f"You excel at role-playing. Picture yourself as user {self.avatar_id} who has just finished exploring a movie recommendation system. You have the following social traits:"
                    +f"\nYour activity trait is described as: {self.activity_dsc}"
                    +f"\nYour conformity trait is described as: {self.conformity_dsc}"
                    +f"\nYour diversity trait is described as: {self.diversity_dsc}"
                    +f"\nBeyond that, your movie tastes are: {'; '.join(self.taste).replace('I ','')}. "
                    +"\nThe activity characteristic pertains to the frequency of your movie-watching habits. The conformity characteristic measures the degree to which your ratings are influenced by historical ratings. The diversity characteristic gauges your likelihood of watching movies that may not align with your usual taste."
                    )
        prompt = f"""
        Relevant context from user {self.avatar_id}'s memory:
        {formated_relevant_memories}
        Act as user {self.avatar_id}, assume you are having a interview, reponse the following question:
        {question}
        """


        messages = [{"role": "system",
                    "content": sys_prompt},
                    {"role": "user",
                    "content": prompt}]
        
        self.write_log("\n" + sys_prompt, color="blue")
        self.write_log("\n" + prompt, color="blue")
        response = self._reaction(
            messages,
            fallback_response=self._fallback_interview_response(),
            context_label="response_to_question",
        )
        self.write_log("\n" + response, color="blue")
        # 
        if(remember):
            self.memory.add_memory(f"I was asked '{question}', and I responsed: '{response}'"
                                , now=datetime.datetime.now())
        return response
    
    def reaction_to_forced_items(self, recommended_items_str):
        """
        Summarize the feelings of the avatar for recommended item list.
        """
        if self.prompt_profile == "beauty":
            sys_prompt = (
                f"You are role-playing as a realistic shopper using a {self.item_word} recommendation system."
                f"\nYour long-term preferences: {'; '.join(self.taste).replace('I ', '')}."
            )
            if self.timeline:
                sys_prompt += f"\nYour chronological interaction history (oldest to newest): {self.timeline}"

            prompt = (
                "#### Candidate List ####\n"
                + recommended_items_str
                + f"\nTask: For each {self.item_word} above, decide whether you would click/try it."
                + "\nIMPORTANT ID RULE: In the MOVIE field, output the numeric ID shown as 'ID: <number>' in the list. Do NOT output titles."
                + "\nOutput exactly one line per item. No extra lines."
                + "\nUse this format: MOVIE: <ID>; WATCH: <yes/no>; REASON: <short reason grounded in the shown info and your preferences>."
            )
        elif self.prompt_profile == "book":
            sys_prompt = (
                "You are role-playing as a realistic reader using a book recommendation system."
                + f"\nYour long-term preferences: {'; '.join(self.taste).replace('I ', '')}."
            )
            if self.timeline:
                sys_prompt += f"\nYour chronological reading history (oldest to newest): {self.timeline}"

            prompt = (
                "#### Candidate List ####\n"
                + recommended_items_str
                + "\nTask: For each book above, decide whether you would open or shortlist it now."
                + "\nIMPORTANT ID RULE: In the MOVIE field, output the numeric ID shown as 'ID: <number>' in the list. Do NOT output titles."
                + "\nOutput exactly one line per item. No extra lines."
                + "\nUse this format: MOVIE: <ID>; WATCH: <yes/no>; REASON: <short reason grounded in shown topic/author/summary and your preferences>."
            )
        elif self.prompt_profile == "short_video":
            sys_prompt = (
                f"You are role-playing as a realistic short-video viewer."
                f"\nYour long-term preferences: {'; '.join(self.taste).replace('I ', '')}."
            )
            if self.timeline:
                sys_prompt += f"\nYour recent watch timeline: {self.timeline}"

            prompt = (
                "#### Candidate List ####\n"
                + recommended_items_str
                + "\nTask: For each short video above, decide whether you would watch it now."
                + "\nIMPORTANT ID RULE: In the MOVIE field, output the numeric ID shown as 'ID: <number>' in the list. Do NOT output titles."
                + "\nOutput exactly one line per item. No extra lines."
                + "\nUse this format: MOVIE: <ID>; WATCH: <yes/no>; REASON: <short reason grounded in shown tags/summary and your preferences>."
            )
        else:
            sys_prompt = ("Assume you are a user browsing movie recommendation system who has the following characteristics: "
                    +f"\nYour movie tastes are: {'; '.join(self.taste).replace('I ','')}. ")
            prompt = (
                    "##recommended list## \n" 
                    +recommended_items_str
                    +"\nPlease choose movies in the ##recommended list## that you want to watch and explain why. After watching the movie, evaluate each movie based on your characteristics, taste and historical ratings to give a rating from 1 to 5."
                    +"\nYou only watch movies which aligh with your taste."
                    +"\nUse this format: MOVIE: [movie name]; WATCH: [yes or no]; REASON: [brief reason]"
                    "\nYou must judge all the movies. If you don't want to watch a movie, use WATCH: no; REASON: [brief reason]"
                    +"\nEach response should be on one line. Do not include any additional information or explanations and stay grounded in reality."
            )
        messages = [{"role": "system",
                    "content": sys_prompt},
                    {"role": "user",
                    "content": prompt}]

        reaction = self._reaction(messages, timeout=60)

        return reaction
    
    def reaction_to_recommended_items(self, recommended_items_str, current_page):
        """
        Summarize the feelings of the avatar for recommended item list.
        """ 
        try:
            high_rating = self.high_rating.replace('You are','')
        except:
            high_rating = ''
        low_rating = self.low_rating

        if self.prompt_profile == "short_video":
            sys_prompt = (
                "You excel at role-playing. Picture yourself as a user exploring a short-video recommendation system."
                + f"\nYour activity trait: {self.activity_dsc}"
                + f"\nYour conformity trait: {self.conformity_dsc}"
                + f"\nYour diversity trait: {self.diversity_dsc}"
                + f"\nYour content taste: {'; '.join(self.taste).replace('I ', '')}"
                + f"\nHigh-rating tendency: {high_rating}"
                + f"\nLow-rating tendency: {low_rating}"
                + "\nStay realistic and conservative: if relevance is weak, do not watch."
            )
            if self.timeline:
                tl = self._compact_timeline_text(self.timeline)
                sys_prompt = sys_prompt + f"\nRecent timeline (oldest->newest, sliced): {tl}"
            if self.memory.memory_retriever.memory_stream:
                observation = "What short videos have you watched on previous pages?"
                relevant_memories = self.memory.fetch_memories(observation)
                formated_relevant_memories = self.memory.format_memories_detail(relevant_memories)
                sys_prompt = sys_prompt + f"\nRelevant context from your memory:{formated_relevant_memories}"

            prompt = (
                "#### Recommended List #### \n"
                + f"PAGE {current_page}\n"
                + recommended_items_str
                + "\nInterpretation in short-video context:"
                + "\n- ALIGN=yes means content relevance is clear."
                + "\n- WATCH means I will really watch this video now."
                + "\n- RATING is expected satisfaction after reading the shown information."
                + "\nIMPORTANT ID RULE: For every line, MOVIE must be the numeric ID shown as 'ID: <number>' on this page. Do not output titles."
                + "\nStep 1: For every item, output exactly one line:"
                + "\nMOVIE: <ID>; ALIGN: <yes/no>; REASON: <brief reason grounded in tags/summary/history>;"
                + "\nStep 2: Decide what to watch now (can be None):"
                + "\nNUM: <N>; WATCH: <ID1 | ID2 | ... or None>; REASON: <brief reason>;"
                + "\nStep 3: For each watched item in WATCH, output one line:"
                + "\nMOVIE: <ID>; RATING: <1-5 integer>; FEELING: <brief aftermath sentence>;"
                + "\nUse full rating scale: 1-2 dislike, 3 neutral, 4 like, 5 strong like."
                + "\nNo extra text beyond required lines."
            )
        elif self.prompt_profile == "book":
            sys_prompt = (
                "You excel at role-playing. Picture yourself as a realistic reader browsing a book recommendation system."
                + f"\nYour activity trait is described as: {self.activity_dsc}"
                + f"\nYour conformity trait is described as: {self.conformity_dsc}"
                + f"\nYour diversity trait is described as: {self.diversity_dsc}"
                + f"\nYour long-term reading taste: {'; '.join(self.taste).replace('I ', '')}"
                + f"\nHigh-rating tendency: {high_rating}"
                + f"\nLow-rating tendency: {low_rating}"
                + "\nStay realistic: only open books that feel meaningfully relevant."
            )
            if self.timeline:
                tl = self._compact_timeline_text(self.timeline)
                sys_prompt = sys_prompt + f"\nRecent reading timeline (oldest->newest, sliced): {tl}"
            if self.memory.memory_retriever.memory_stream:
                observation = "What books have you opened or liked on previous pages?"
                relevant_memories = self.memory.fetch_memories(observation)
                formated_relevant_memories = self.memory.format_memories_detail(relevant_memories)
                sys_prompt = sys_prompt + f"\nRelevant context from your memory:{formated_relevant_memories}"

            prompt = (
                "#### Recommended List #### \n"
                + f"PAGE {current_page}\n"
                + recommended_items_str
                + "\nInterpretation in a book-browsing context:"
                + "\n- ALIGN=yes means this book clearly fits my current interests and is worth opening."
                + "\n- WATCH means I will click/open details or shortlist this book now."
                + "\n- RATING is expected reading satisfaction if I choose this book."
                + "\nIMPORTANT ID RULE: For every line, MOVIE must be the numeric ID shown as 'ID: <number>' on this page. Do not output titles."
                + "\nStep 1: For every item, output exactly one line:"
                + "\nMOVIE: <ID>; ALIGN: <yes/no>; REASON: <brief reason grounded in title/tags/summary/history>;"
                + "\nConservative alignment rule: use ALIGN=yes only for books with clear topic/author/genre fit."
                + "\nStep 2: Decide what to open now (can be None):"
                + "\nNUM: <N>; WATCH: <ID1 | ID2 | ... or None>; REASON: <brief reason>;"
                + "\nConservative watch rule: if the fit is weak or uncertain, output WATCH: None."
                + "\nStep 3: For each watched item in WATCH, output one line:"
                + "\nMOVIE: <ID>; RATING: <1-5 integer>; FEELING: <mention interest, reading value, and any concern about fit>;"
                + "\nUse full rating scale: 1-2 poor match, 3 neutral/unsure, 4 good fit, 5 strong fit."
                + "\nNo extra text beyond required lines."
            )
        elif self.use_beauty_prompt:
            beauty_mode = str(getattr(self.args, "beauty_prompt_mode", os.getenv("BEAUTY_PROMPT_MODE", "a"))).strip().lower()
            if beauty_mode not in {"a", "b", "c"}:
                beauty_mode = "a"

            def parse_page_items(text: str):
                # Parse the "<- ID: ... -> <- Title: ... -> <- Avg rating: ... -> <- Tags: ... -> <- Summary: ... ->" format.
                items = []
                if not text:
                    return items
                block_re = re.compile(
                    r"<-\s*ID:\s*(\d+)\s*->\s*<-\s*Title:\s*(.*?)\s*->\s*<-\s*Avg rating:\s*(.*?)\s*->\s*<-\s*Tags:\s*(.*?)\s*->\s*<-\s*Summary:\s*(.*?)\s*->",
                    re.IGNORECASE | re.DOTALL,
                )
                for m in block_re.finditer(text):
                    try:
                        iid = int(m.group(1))
                    except Exception:
                        continue
                    title = self._clean_optional_text(m.group(2))
                    avg = self._clean_optional_text(m.group(3))
                    tags = self._clean_optional_text(m.group(4))
                    summary = self._clean_optional_text(m.group(5))
                    items.append({"id": iid, "title": title, "avg": avg, "tags": tags, "summary": summary})
                return items

            def extract_review_snippets(summary: str):
                # Heuristic: snippets like "2.0/5 text; 5.0/5 text"
                pos, neg, neutral = [], [], []
                if not summary:
                    return pos, neg, neutral
                for r, txt in re.findall(r"(\d+(?:\.\d+)?)/5\s*([^;]+)", summary):
                    try:
                        score = float(r)
                    except Exception:
                        score = 0.0
                    snippet = self._clean_optional_text(txt)
                    if not snippet:
                        continue
                    if score >= 4.0:
                        pos.append(snippet)
                    elif score <= 2.0:
                        neg.append(snippet)
                    else:
                        neutral.append(snippet)
                return pos[:2], neg[:2], neutral[:2]

            def parse_review_count(summary: str):
                m = re.search(r"review\s*count\s*:\s*(\d+)", summary or "", flags=re.IGNORECASE)
                if not m:
                    return None
                try:
                    return int(m.group(1))
                except Exception:
                    return None

            def parse_price(summary: str):
                m = re.search(r"price\s*:\s*\$?\s*([0-9]+(?:\.[0-9]+)?)", summary or "", flags=re.IGNORECASE)
                if not m:
                    return None
                return m.group(1).strip()

            def build_detail_cards(page_items, shortlist_ids):
                by_id = {it["id"]: it for it in page_items}
                cards = []
                for iid in shortlist_ids:
                    it = by_id.get(iid)
                    if not it:
                        continue
                    pos, neg, neutral = extract_review_snippets(it.get("summary", ""))
                    pros = "; ".join(pos) if pos else "N/A"
                    cons = "; ".join(neg) if neg else ("; ".join(neutral) if neutral else "N/A")
                    review_count = parse_review_count(it.get("summary", ""))
                    price = parse_price(it.get("summary", ""))
                    cards.append(
                        "DETAIL CARD\n"
                        + f"ID: {iid}\n"
                        + f"Title: {it.get('title','')}\n"
                        + f"Tags: {it.get('tags','')}\n"
                        + f"Avg rating: {it.get('avg','')}\n"
                        + f"Review count: {review_count if review_count is not None else 'N/A'}\n"
                        + f"Price: {price if price else 'N/A'}\n"
                        + f"Pros signal: {pros}\n"
                        + f"Cons/Risk signal: {cons}\n"
                        + f"Summary: {it.get('summary','')}\n"
                    )
                return "\n".join(cards).strip()

            sys_prompt = (f"You excel at role-playing. Picture yourself as a user exploring a {self.item_word} recommendation system. You have the following social traits:"
                    +f"\nYour activity trait is described as: {self.activity_dsc}"
                    +f"\nYour conformity trait is described as: {self.conformity_dsc}"
                    +f"\nYour diversity trait is described as: {self.diversity_dsc}"
                    +f"\nBeyond that, your {self.item_word} tastes are: {'; '.join(self.taste).replace('I ','')}. "
                    +f"\nYour high-rating tendency: {high_rating}"
                    +f"\nYour low-rating tendency: {low_rating}"
                    +f"\nThe activity characteristic pertains to the frequency of your {self.item_word} browsing and consumption habits. The conformity characteristic measures the degree to which your ratings are influenced by historical ratings. The diversity characteristic gauges your likelihood of trying {self.item_word}s that may not align with your usual taste."
                    )
            if self.timeline:
                tl = self._compact_timeline_text(self.timeline)
                sys_prompt = sys_prompt + f"\nYour chronological interaction history (oldest to newest, recent slice): {tl}"
            if self.memory.memory_retriever.memory_stream:
                observation = f"What {self.item_word}s have you chosen on previous pages of the current recommender system?"
                relevant_memories = self.memory.fetch_memories(observation)
                formated_relevant_memories = self.memory.format_memories_detail(relevant_memories)
                sys_prompt = sys_prompt +f"\nRelevant context from your memory:{formated_relevant_memories}"

            if beauty_mode == "c" and self.session_profile:
                sp = self.session_profile
                sys_prompt = (
                    sys_prompt
                    + "\n\nSESSION INTENT (fixed for this whole session):"
                    + f"\n- Mission: {str(sp.get('mission','restock')).upper()} (time-limited, avoid misclicks)"
                    + f"\n- Target category: {sp.get('focus','') or 'N/A'}"
                    + "\n\nSTRICT CLICK POLICY:"
                    + "\n- Default is WATCH: None. Clicking is costly; only click when the item is a clear, low-risk fit."
                    + f"\n- To WATCH (click): category match MUST be obvious AND Avg rating >= {sp.get('watch_rating_min', 4.2)}"
                    + f" AND Review count >= {sp.get('watch_review_min', 120)}."
                    + f"\n- To ALIGN=yes: category match MUST be obvious AND Avg rating >= {sp.get('align_rating_min', 4.0)}"
                    + f" AND Review count >= {sp.get('align_review_min', 60)}."
                    + "\n- Red flags: any <=2/5 snippet, irritation/allergy, leaking/broken, strong negative wording => do NOT WATCH."
                    + "\n- If key info is missing for a risky product (actives/skin issues), be conservative and do NOT WATCH."
                    + f"\n- Calibration: you usually WATCH on <= {int(round(float(sp.get('target_watch_rate', 0.10)) * 100))}% of pages."
                )

            if beauty_mode == "c":
                page_items = parse_page_items(recommended_items_str)
                candidate_ids = [it["id"] for it in page_items]
                detail_cards = build_detail_cards(page_items, candidate_ids) if candidate_ids else ""
                sp = self.session_profile or {}
                prompt = (
                    "#### Page Items (parsed cards) ####\n"
                    + (detail_cards + "\n\n" if detail_cards else "")
                    + "You are browsing in a realistic shopping session. Stay grounded in the information shown on this page and your session intent.\n"
                    + "IMPORTANT ID RULE: For every line you output, the MOVIE field must be the numeric ID shown as 'ID: <number>' on this page. Do NOT output titles in MOVIE. Only use IDs from this page.\n"
                    + "FORMAT RULES: One response per line. No extra lines. Use exact punctuation and keywords.\n"
                    + "Interpretation in a shopping context:\n"
                    + "- ALIGN=yes means: worth clicking into details (potential fit).\n"
                    + "- WATCH means: you will actually click/view details now (rare; only for clear, low-risk matches).\n"
                    + "- RATING is expected satisfaction after reading available details.\n"
                    + f"Hard rule: WATCH at most {int(sp.get('max_watch', 1))} item(s) on this page.\n"
                    + "Step 1 (Fit): For each item on this page, output one line:\n"
                    + "MOVIE: <ID>; ALIGN: <yes/no>; REASON: <cite category match + rating/review evidence or a concrete risk>.\n"
                    + "Step 2 (Click decision): Output exactly one line:\n"
                    + "NUM: <N>; WATCH: <ID1 | ID2 | ... or None>; REASON: <brief reason>;\n"
                    + "Step 3 (Expected satisfaction): For each chosen item in WATCH, output one line:\n"
                    + "MOVIE: <ID>; RATING: <1-5 integer>; FEELING: <expected effect + risk/value>.\n"
                )
            else:
                prompt = (
                    "#### Recommended List #### \n"
                    + f"PAGE {current_page}\n"
                    + recommended_items_str
                    + "\nYou are browsing in a realistic shopping session. Stay grounded in the information shown on this page (ID/Title/Avg rating/Tags/Summary) and your history."
                    + "\nWhen your long-term taste conflicts with the timeline, prioritize recent timeline signals."
                    + "\nIMPORTANT ID RULE: For every line you output, the MOVIE field must be the numeric ID shown as 'ID: <number>' on this page. Do NOT output titles in MOVIE. Only use IDs from this page."
                    + "\nFORMAT RULES: One response per line. No extra lines. Use exact punctuation and keywords."
                    + "\nInterpretation in a shopping context:"
                    + "\n- ALIGN=yes means: this item fits my current needs and is worth clicking into details."
                    + "\n- WATCH means: I will click/view details and seriously consider adding to cart/wishlist on this page."
                    + "\n- RATING is my expected satisfaction after reading the available details (not a guaranteed post-purchase rating)."
                    + "\nStep 1 (Fit & click-worthiness): For each item on this page, output one line:"
                    + "\nMOVIE: <ID>; ALIGN: <yes/no>; REASON: <1 short reason grounded in Tags/Summary/history, mention fit OR a concrete risk>."
                    + "\nConservative alignment rule: use ALIGN=yes only for clear matches; if uncertain/weakly relevant, use ALIGN=no."
                    + "\nStep 2 (Action on this page): From ALIGN=yes items, decide what you will click/shortlist now, based on your activity/diversity traits."
                    + "\nNUM: <N>; WATCH: <ID1 | ID2 | ... or None>; REASON: <brief reason>;"
                    + "\nConservative watch rule: if information is weak or risks are high, output WATCH: None."
                    + "\nStep 3 (Expected satisfaction): For each chosen item in WATCH, output one line:"
                    + "\nMOVIE: <ID>; RATING: <1-5 integer>; FEELING: <include at least 2 of: expected effect, risk flag, value/price; and state whether you follow Avg rating or your own preference>."
                    + "\nRating calibration rule: use the full scale. 1-2 = poor match/avoid, 3 = unsure/neutral, 4 = likely good, 5 = strong buy."
                )
        else:
            sys_prompt = ("You excel at role-playing. Picture yourself as a user exploring a movie recommendation system. You have the following social traits:"
                    +f"\nYour activity trait is described as: {self.activity_dsc}"
                    +f"\nYour conformity trait is described as: {self.conformity_dsc}"
                    +f"\nYour diversity trait is described as: {self.diversity_dsc}"
                    +f"\nBeyond that, your movie tastes are: {'; '.join(self.taste).replace('I ','')}. "
                    +f"\nAnd your rating tendency is {high_rating}"
                    +"\nThe activity characteristic pertains to the frequency of your movie-watching habits. The conformity characteristic measures the degree to which your ratings are influenced by historical ratings. The diversity characteristic gauges your likelihood of watching movies that may not align with your usual taste."
                    )
            if self.memory.memory_retriever.memory_stream:
                observation = "What movies have you watched on the previous pages of the current recommender system?"
                relevant_memories = self.memory.fetch_memories(observation)
                formated_relevant_memories = self.memory.format_memories_detail(relevant_memories)
                sys_prompt = sys_prompt +f"\nRelevant context from your memory:{formated_relevant_memories}"

            prompt = (
                    "#### Recommended List #### \n"
                    + f"PAGE {current_page}\n"
                    +recommended_items_str
                    +"\nPlease respond to all the movies in the ## Recommended List ## and provide explanations."
                    +"\nFirstly, determine which movies align with your taste and which do not, and provide reasons. You must respond to all the recommended movies using this format:"
                    +"\nMOVIE: [movie name]; ALIGN: [yes or no]; REASON: [brief reason]"
                    +"\nConservative alignment rule: use ALIGN=yes only when there is clear evidence this movie fits your taste."
                    +"\nSecondly, among the movies that align with your tastes, decide the number of movies you want to watch based on your activity and diversity traits. Use this format:"
                    +"\nNUM: [number of movie you choose to watch]; WATCH: [all movie name you choose to watch]; REASON: [brief reason];"
                    +"\nConservative watch rule: if none is strongly attractive, choose WATCH: None."
                    +"\nThirdly, assume it's your first time watching the movies you've chosen, and rate them on a scale of 1-5 to reflect different degrees of liking, considering your feeling and conformity trait. Use this format:"
                    +"\n MOVIE:[movie you choose to watch]; RATING: [integer between 1-5]; FEELING: [aftermath sentence]; "
                    +"\nUse full rating scale: 1-2 means dislike/poor match, 3 means neutral, 4 means like, 5 means strong like."
                    +"\n Do not include any additional information or explanations and stay grounded."
            )

        if self.prompt_profile == "beauty" and beauty_mode == "b":
            page_items = parse_page_items(recommended_items_str)
            candidate_ids = [it["id"] for it in page_items]
            # Stage 1: SCAN (shortlist + ask for info)
            scan_prompt = (
                "SCAN STAGE (fast skim, do not decide rating yet).\n"
                + "Given the page items, pick up to 2 IDs to inspect deeper. If none, use None.\n"
                + "Output exactly 2 lines:\n"
                + "SHORTLIST: <ID1 | ID2 | ... or None>;\n"
                + "INFO_NEEDED: <what missing info would change your decision (e.g., ingredients, skin type, scent, durability, size)>;"
            )
            scan_messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": scan_prompt}]
            self.write_log("\n[BeautyMode-B][SCAN][SYS]\n" + sys_prompt, color="blue")
            self.write_log("\n[BeautyMode-B][SCAN][USER]\n" + scan_prompt, color="blue")
            scan_out = self._reaction(
                scan_messages,
                timeout=60,
                fallback_response=self._fallback_scan_response(),
                context_label="beauty_mode_b_scan",
            )
            self.write_log("\n[BeautyMode-B][SCAN][OUT]\n" + scan_out, color="yellow")

            shortlist_ids = []
            m = re.search(r"^\s*SHORTLIST\s*:\s*(.*?)\s*;", scan_out or "", flags=re.IGNORECASE | re.MULTILINE)
            if m:
                raw = m.group(1).strip()
                if raw.lower() not in {"none", "null", "n/a", "[]"}:
                    shortlist_ids = [int(x) for x in re.findall(r"\d+", raw)]
            shortlist_ids = [iid for iid in shortlist_ids if iid in set(candidate_ids)]
            shortlist_ids = shortlist_ids[:2]

            info_needed = ""
            m2 = re.search(r"^\s*INFO_NEEDED\s*:\s*(.*?)\s*;", scan_out or "", flags=re.IGNORECASE | re.MULTILINE)
            if m2:
                info_needed = self._clean_optional_text(m2.group(1))

            detail_cards = build_detail_cards(page_items, shortlist_ids)
            decide_sys_prompt = (
                sys_prompt
                + "\n\nDECIDE STAGE.\n"
                + f"Shortlist from SCAN: {shortlist_ids if shortlist_ids else 'None'}\n"
                + (f"Info you wanted: {info_needed}\n" if info_needed else "")
                + ("Available detail cards below. If info is still missing, be conservative.\n\n" + detail_cards if detail_cards else "\nNo extra details available; decide from the page only.\n")
            )
            decide_messages = [{"role": "system", "content": decide_sys_prompt}, {"role": "user", "content": prompt}]
            self.write_log("\n[BeautyMode-B][DECIDE][SYS]\n" + decide_sys_prompt, color="blue")
            self.write_log("\n[BeautyMode-B][DECIDE][USER]\n" + prompt, color="blue")
            reaction = self._reaction(
                decide_messages,
                timeout=60,
                fallback_response=self._fallback_recommended_items_response(recommended_items_str),
                context_label="beauty_mode_b_decide",
            )
            self.write_log("\n" + reaction, color="yellow")
        else:
            messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}]
            self.write_log("\n" + sys_prompt, color="blue")
            self.write_log("\n" + prompt, color="blue")
            reaction = self._reaction(
                messages,
                timeout=60,
                fallback_response=self._fallback_recommended_items_response(recommended_items_str),
                context_label="reaction_to_recommended_items",
            )  # reaction
            self.write_log("\n" + reaction, color="yellow")

        # @ 2 Add user satisfaction information for this page.

        # =========================
        # Be tolerant to spacing variants like "MOVIE:Title" and "MOVIE: Title".
        pattern1 = re.compile(
            r'^\s*MOVIE\s*:\s*(.+?)\s*;\s*RATING\s*:\s*(\d+)\s*;\s*FEELING\s*:\s*(.*?)\s*$',
            re.IGNORECASE | re.MULTILINE,
        )
        match1 = pattern1.findall(reaction)
        pattern2 = re.compile(
            r'^\s*MOVIE\s*:\s*(.+?)\s*;\s*ALIGN\s*:\s*(.+?)\s*;\s*REASON\s*:\s*(.*?)\s*$',
            re.IGNORECASE | re.MULTILINE,
        )
        match2 = pattern2.findall(reaction)
        id_title_map = self._extract_page_id_title_map(recommended_items_str)
        match1 = [
            (self._normalize_page_token(movie_title, id_title_map), rating, feeling)
            for movie_title, rating, feeling in match1
        ]
        match2 = [
            (self._normalize_page_token(movie_title, id_title_map), align, reason)
            for movie_title, align, reason in match2
        ]
        all_movies = ", ".join(self._dedupe_text_keep_order([movie_title for movie_title, align, reason in match2]))
        watched_movies = self._dedupe_text_keep_order([movie_title for movie_title, rating, feeling in match1])
        watched_movies_ratings = [rating.strip(';') for movie_title, rating, feeling in match1]
        like_movies = self._dedupe_text_keep_order(
            [movie_title for movie_title, rating, feeling in match1 if int(rating.strip(';')) == 5]
        )
        dislike_movies = [movie_title for movie_title, rating, feeling in match1 if (int(rating.strip(';')) < 4)]
        dislike_movies.extend([movie_title for movie_title, align, reason in match2 if align.strip(';').lower() == 'no'])
        dislike_movies = self._dedupe_text_keep_order(dislike_movies)
        if self.prompt_profile in {"beauty", "short_video", "book"}:
            self.memory.add_memory(f"The recommender recommended the following items to me on page {current_page}: {all_movies}, among them, I selected {watched_movies} and rated them {watched_movies_ratings} respectively. I disliked the rest items: {dislike_movies}."
                , now=datetime.datetime.now()
            )
        else:
            self.memory.add_memory(f"The recommender recommended the following movies to me on page {current_page}: {all_movies}, among them, I watched {watched_movies} and rate them {watched_movies_ratings} respectively. I dislike the rest movies: {dislike_movies}."
                , now=datetime.datetime.now()
            )

        # User makes the next decision.
        next_decision = self.make_next_decision(current_page=current_page)
        if self._is_negative_decision(next_decision):
            self.negative_feedback_count += 1

        if self.negative_feedback_count >= self.exit_threshold:
            self.exit_flag = True
            self.memory.add_memory(f"After browsing {current_page} pages, accumulated negative feedback reached {self.negative_feedback_count}, so I decided to leave the recommendation system."
                , now=datetime.datetime.now())
        
        else:
            self.memory.add_memory(f"Turn to page {current_page+1} of the recommendation. Current accumulated negative feedback: {self.negative_feedback_count}."
                , now=datetime.datetime.now())
        #===========================

        return reaction

    def write_log(self, log, color=None, attrs=None, print=False):
        with open(self.log_file, 'a', encoding='utf-8', errors='replace') as f:
            f.write(str(log) + '\n')
            f.flush()
        if(print):
            cprint(log, color=color, attrs=attrs)
