from simulation.avatar import Avatar
from simulation.base.abstract_arena import abstract_arena
from simulation.hazard_plan import HazardPlanReranker
from termcolor import colored, cprint
import pandas as pd
import os
import os.path as op
import json
from pathlib import Path

import time
import re
import numpy as np
import pickle

import simulation.vars as vars
from simulation.utils import *

class Arena(abstract_arena):
    def __init__(self, args):
        super().__init__(args)
        
        self.max_pages = args.max_pages
        self.finished_num = 0
        self.hazard_plan_enabled = bool(getattr(args, "enable_hazard_plan", False))
        self.hazard_plan_reranker = None
        dataset_name = str(self.dataset).lower()
        self.use_beauty_prompt = (
            ("beauty" in dataset_name)
            or ("amazon" in dataset_name)
            or ("kuairand" in dataset_name)
            or ("book" in dataset_name)
        )

    def load_additional_info(self):
        
        self.user_profile_csv = pd.read_csv(f'datasets/{self.dataset}/raw_data/agg_top_25.csv')

        # return super().load_additional_info()
        self.add_advert = self.args.add_advert
        self.display_advert = self.args.display_advert
        if(self.add_advert):
            self.total_adverts, self.clicked_adverts = 0, 0
            advert_pool = pd.read_pickle(f'datasets/{self.dataset}/simulation/advertisement_review.pkl')
            advert_dict = {'all': {**advert_pool['pop_high_rating'], **advert_pool['pop_low_rating'], **advert_pool['unpop_high_rating'], **advert_pool['unpop_low_rating']}, 
                        'pop_high':advert_pool['pop_high_rating'], 'pop_low':advert_pool['pop_low_rating'], 'unpop_high':advert_pool['unpop_high_rating'], 'unpop_low':advert_pool['unpop_low_rating']}
            # print(self.args.advert_type)
            self.advert = advert_dict[self.args.advert_type]
            self.advert_word = "The best movie you should not miss in your life! "

    def initialize_all_avatars(self):
        """
        initialize avatars
        """
        super().initialize_all_avatars()
        # self.persona_df = pd.read_csv(f"datasets/{self.dataset}/simulation/all_personas_like_information_house.csv")
        self.persona_df = pd.read_csv(f"datasets/{self.dataset}/simulation/all_personas_like_modify.csv")
        self.user_statistic = pd.read_csv(f'datasets/{self.dataset}/simulation/user_statistic.csv', index_col=0)
        # @ avatars and evaluation indicators
        self.avatars = {}
        self.ratings = {}
        self.new_train_dict = {}
        self.exit_page = {}
        self.perf_per_page = {}
        self.watch = {}
        self.n_likes = {}
        self.remaining_users = list(range(self.n_avatars))

        for avatar_id in self.simulated_avatars_id:
            self.avatars[avatar_id] = Avatar(self.args, avatar_id, self.persona_df.loc[avatar_id], self.user_statistic.loc[avatar_id])
            self.new_train_dict[avatar_id] = self.data.train_user_list[avatar_id]
            self.ratings[avatar_id] = []
            self.n_likes[avatar_id] = []
            self.watch[avatar_id] = []
            self.exit_page[avatar_id] = 0
            self.perf_per_page[avatar_id] = []
        self._maybe_init_hazard_plan()

    def _maybe_init_hazard_plan(self):
        if not self.hazard_plan_enabled:
            self.hazard_plan_reranker = None
            return
        artifact_dir = Path("recommenders") / "weights" / self.dataset / "HazardPlan" / str(
            getattr(self.args, "hazard_plan_dir", "Saved")
        )
        if not artifact_dir.exists():
            print(f"[HazardPlan] artifact dir not found, fallback to heuristic reranking: {artifact_dir}")
        self.hazard_plan_reranker = HazardPlanReranker(
            dataset=self.dataset,
            movie_detail=self.movie_detail,
            persona_df=self.persona_df,
            user_statistic=self.user_statistic,
            artifact_dir=artifact_dir if artifact_dir.exists() else None,
            candidate_pool=int(getattr(self.args, "hazard_plan_candidate_pool", 50)),
            override_plan=str(getattr(self.args, "hazard_plan_override", "auto")),
        )
        print(
            "[HazardPlan] enabled with "
            + (
                f"learned models from {artifact_dir}"
                if artifact_dir.exists() and getattr(self.hazard_plan_reranker, "model", None) is not None
                else "heuristic exit-risk fallback"
            )
        )
    
    def page_generator(self, avatar_id):
        """
        generate one page items for one avatar
        """
        if not self.hazard_plan_enabled or self.hazard_plan_reranker is None:
            i = 0
            while (i+1)*self.items_per_page < self.data.n_items:
                yield self.full_rankings[avatar_id][i*self.items_per_page:(i+1)*self.items_per_page]
                i += 1
            return

        remaining = [int(x) for x in np.asarray(self.full_rankings[avatar_id]).tolist()]
        page_index = 0
        while remaining:
            page_index += 1
            candidate_pool = remaining[: max(int(self.args.hazard_plan_candidate_pool), self.items_per_page)]
            selected_ids, debug_meta = self.hazard_plan_reranker.score_candidates(
                arena=self,
                avatar_id=avatar_id,
                candidate_ids=candidate_pool,
                page_index=page_index,
                items_per_page=self.items_per_page,
            )
            if not selected_ids:
                selected_ids = candidate_pool[: self.items_per_page]
            selected_ids = [int(x) for x in selected_ids[: self.items_per_page]]
            if not selected_ids:
                break
            for item_id in selected_ids:
                try:
                    remaining.remove(int(item_id))
                except ValueError:
                    pass
            avatar_ = self.avatars.get(avatar_id)
            if avatar_ is not None:
                avatar_.write_log(
                    f"[HazardPlan] PAGE {page_index} plan={debug_meta.get('plan')} "
                    + f"selected={selected_ids} top={debug_meta.get('top_candidates', [])}"
                )
            yield np.asarray(selected_ids, dtype=np.int64)

    @staticmethod
    def _normalize_field(value):
        if value is None:
            return ""
        return re.sub(r"\s+", " ", str(value)).strip().strip(";").strip()

    @staticmethod
    def _safe_console_print(value):
        try:
            print(value)
        except UnicodeEncodeError:
            try:
                safe_value = str(value).encode("utf-8", errors="replace").decode(
                    "utf-8", errors="replace"
                )
                print(safe_value)
            except Exception:
                print("[Arena] non-encodable output skipped")

    def _split_watch_titles(self, watch_text, candidate_titles):
        watch_text = self._normalize_field(watch_text)
        if not watch_text:
            return []
        if watch_text.lower() in {"none", "no", "n/a", "null", "[]"}:
            return []

        candidate_titles = candidate_titles or []
        if candidate_titles:
            lowered_watch = watch_text.lower()
            matched = []
            for title in sorted(candidate_titles, key=len, reverse=True):
                if title.lower() in lowered_watch:
                    matched.append(title)
            if matched:
                matched_set = set(matched)
                return [title for title in candidate_titles if title in matched_set]

        split_tokens = re.split(r"\s*(?:\||/|;|\band\b)\s*", watch_text, flags=re.IGNORECASE)
        return [self._normalize_field(token) for token in split_tokens if self._normalize_field(token)]

    def _parse_reaction_output(self, response, candidate_titles=None):
        text = response or ""
        align_entries = []
        rating_entries = []
        watch_titles = []
        watch_reason = ""

        align_pattern = re.compile(
            r"^\s*MOVIE\s*:\s*(.+?)\s*;\s*ALIGN\s*:\s*([^;\n]+)\s*;\s*REASON\s*:\s*(.*?)\s*$",
            re.IGNORECASE | re.MULTILINE,
        )
        rating_pattern = re.compile(
            r"^\s*MOVIE\s*:\s*(.+?)\s*;\s*RATING\s*:\s*(\d+)\s*;\s*FEELING\s*:\s*(.*?)\s*$",
            re.IGNORECASE | re.MULTILINE,
        )
        watch_pattern = re.compile(
            r"^\s*NUM\s*:\s*\d+\s*;\s*WATCH\s*:\s*(.*?)\s*;\s*REASON\s*:\s*(.*?)\s*;?\s*$",
            re.IGNORECASE | re.MULTILINE,
        )
        watch_pattern_loose = re.compile(
            r"^\s*WATCH\s*:\s*(.*?)\s*;\s*REASON\s*:\s*(.*?)\s*;?\s*$",
            re.IGNORECASE | re.MULTILINE,
        )

        for match in align_pattern.finditer(text):
            align_entries.append(
                {
                    "title": self._normalize_field(match.group(1)),
                    "align": self._normalize_field(match.group(2)).lower(),
                    "reason": self._normalize_field(match.group(3)),
                }
            )

        for match in rating_pattern.finditer(text):
            rating_raw = self._normalize_field(match.group(2))
            try:
                rating_num = int(rating_raw)
            except ValueError:
                continue
            rating_entries.append(
                {
                    "title": self._normalize_field(match.group(1)),
                    "rating": rating_num,
                    "feeling": self._normalize_field(match.group(3)),
                }
            )

        watch_match = watch_pattern.search(text)
        if not watch_match:
            watch_match = watch_pattern_loose.search(text)
        if watch_match:
            watch_titles = self._split_watch_titles(watch_match.group(1), candidate_titles)
            watch_reason = self._normalize_field(watch_match.group(2))

        watch_titles = list(dict.fromkeys([self._normalize_field(title) for title in watch_titles if title]))

        return {
            "align_entries": align_entries,
            "rating_entries": rating_entries,
            "watch_titles": watch_titles,
            "watch_reason": watch_reason,
        }

    def validate_all_avatars(self):
        vars.global_start_time = time.time()
        print("global start time", vars.global_start_time)
        self.precision_list = []
        self.recall_list = []
        self.accuracy_list = []
        self.f1_list = []
        self.start_time = time.time()

        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        loop = asyncio.get_event_loop()
        val_workers = int(os.getenv("ARENA_VAL_MAX_WORKERS", "32"))
        val_workers = max(1, min(val_workers, len(self.simulated_avatars_id)))
        print(f"[Arena] validation parallel workers={val_workers}")
        executor = ThreadPoolExecutor(max_workers=val_workers)
        tasks = []

        t1 = time.time()
        for avatar_id in self.simulated_avatars_id:
            tasks.append(self.async_validate_one_avatar(avatar_id, loop, executor))
        loop.run_until_complete(asyncio.wait(tasks))
        t2 = time.time()
        print(f"Time cost: {t2-t1}s")

        print("precision_list", self.precision_list)
        print("recall_list", self.recall_list)
        print("accuracy_list", self.accuracy_list)
        print("f1_list", self.f1_list)

        with open(self.storage_base_path + "/validation_metrics.txt", 'w') as f:
            f.write(f"Total simulation time: {round(time.time() - self.start_time, 2)}s\n")
            f.write(f"n_avatars: {self.n_avatars}\n")
            f.write(f"Average precision: {np.mean(self.precision_list)}\n")
            f.write(f"Average recall: {np.mean(self.recall_list)}\n")
            f.write(f"Average accuracy: {np.mean(self.accuracy_list)}\n")
            f.write(f"Average f1: {np.mean(self.f1_list)}\n")

    async def async_validate_one_avatar(self, avatar_id, loop, executor):
        """
        async
        validate the effectiveness of the model for one avatar
        avatar_id: the id of the simulated avatar
        """
        avatar_ = self.avatars[avatar_id]
        train_list, val_list, test_list = self.data.train_user_list[avatar_id], self.data.valid_user_list[avatar_id], self.data.test_user_list[avatar_id]

        # Take the union for calculating precision.
        all_items = list(range(self.data.n_items))
        observed_items = list(set(train_list) | set(val_list) | set(test_list))
        selection_candidates = list(set(val_list) | set(test_list))
        unobserved_items = list(set(all_items) - set(observed_items))
        # Pick 5 randomly from the test_list.
        min_val = min(len(selection_candidates), 20//(self.val_ratio+1))
        print(len(selection_candidates), 10)

        test_observed_items = np.random.choice(selection_candidates, min_val, replace=False)
        test_unobserved_items = np.random.choice(unobserved_items, int(min_val*self.val_ratio), replace=False)

        print("test_all", test_observed_items, test_unobserved_items)

        forced_items_ids = np.concatenate((test_observed_items, test_unobserved_items))
        # Randomly shuffle.
        np.random.shuffle(forced_items_ids)

        print("forced_items_ids", forced_items_ids)

        forced_items = [self.movie_detail.loc[idx] for idx in forced_items_ids]

        truth_tmp = [self.movie_detail.loc[idx] for idx in test_observed_items]
        if self.use_beauty_prompt:
            truth_list = []
            for idx, item in zip(test_observed_items, truth_tmp):
                tags = [t for t in str(getattr(item, "genres", "")).split("|") if t]
                tags_str = ", ".join(tags[:6]) if tags else "N/A"
                truth_list.append(
                    "<- ID: " + str(int(idx)) + " ->"
                    + " <- Title: " + str(item.title) + " ->"
                    + " <- Avg rating: " + str(round(float(item.rating), 2)) + " ->"
                    + " <- Tags: " + tags_str + " ->"
                    + " <- Summary: " + str(item.summary) + " ->" + "\n"
                )
        else:
            truth_list = ["<- " + item.title + " ->" 
                                + " <- History ratings:" + str(round(item.rating, 2)) + " ->" 
                                + " <- Summary:" + item.summary + " ->" + "\n"
                                for item in truth_tmp]
        truth_str = ''.join(truth_list)
        cprint(truth_str, color='white', attrs=['bold'])

        if self.use_beauty_prompt:
            recommended_items = []
            for idx, item in zip(forced_items_ids, forced_items):
                tags = [t for t in str(getattr(item, "genres", "")).split("|") if t]
                tags_str = ", ".join(tags[:6]) if tags else "N/A"
                recommended_items.append(
                    "<- ID: " + str(int(idx)) + " ->"
                    + " <- Title: " + str(item.title) + " ->"
                    + " <- Avg rating: " + str(round(float(item.rating), 2)) + " ->"
                    + " <- Tags: " + tags_str + " ->"
                    + " <- Summary: " + str(item.summary) + " ->" + "\n"
                )
        else:
            recommended_items = ["<- " + item.title + " ->" 
                                + " <- History ratings:" + str(round(item.rating, 2)) + " ->" 
                                + " <- Summary:" + item.summary + " ->" + "\n"
                                for item in forced_items]
        recommended_items_str = ''.join(recommended_items)

        response = await loop.run_in_executor(executor, avatar_.reaction_to_forced_items, recommended_items_str)

        cprint(response, color='yellow', attrs=None)

        pattern = re.compile(
            r"^\s*MOVIE\s*:\s*(.+?)\s*;?\s*WATCH\s*:\s*([^;\n]+)\s*;?\s*REASON\s*:\s*(.*?)\s*$",
            re.IGNORECASE | re.MULTILINE,
        )
        matches = pattern.findall(response or "")

        title_id_dict = dict(zip(self.movie_detail["title"], self.movie_detail["movie_id"]))
        title_id_dict_lower = {str(k).lower(): v for k, v in title_id_dict.items()}

        def resolve_item_id(token):
            token = self._normalize_field(token)
            if not token:
                return None
            if token.isdigit():
                return int(token)
            # Exact title match
            if token in title_id_dict:
                try:
                    return int(title_id_dict[token])
                except Exception:
                    return None
            # Case-insensitive title match
            mapped = title_id_dict_lower.get(token.lower())
            if mapped is None:
                return None
            try:
                return int(mapped)
            except Exception:
                return None

        forced_set = set(int(x) for x in forced_items_ids)
        like_movies_ids = set()
        for token, watch, _reason in matches:
            watch_norm = self._normalize_field(watch).lower()
            if watch_norm in {"yes", "y", "true", "1"} or watch_norm.startswith("y"):
                iid = resolve_item_id(token)
                if iid is not None and iid in forced_set:
                    like_movies_ids.add(int(iid))

        print("like_movies_ids", sorted(like_movies_ids))

        pred = np.array([1 if idx in like_movies_ids else 0 for idx in forced_items_ids])
        true = np.array([1 if idx in test_observed_items else 0 for idx in forced_items_ids])

        # Calculate precision.
        precision = get_precision(true, pred)
        print("precision", precision)
        # Calculate recall.
        recall = get_recall(true, pred)
        print("recall", recall)
        accuracy = get_accuracy(true, pred)
        print("accuracy", accuracy)
        f1 = get_f1(true, pred)
        print("f1", f1)

        self.precision_list.append(precision)
        self.recall_list.append(recall)
        self.accuracy_list.append(accuracy)
        self.f1_list.append(f1)

        vars.global_finished_users += 1

    def simulate_all_avatars(self):
        """
        excute the simulation for all avatars
        """
        vars.global_start_time = time.time()
        print("global start time", vars.global_start_time)
        self.start_time = time.time()
        if(self.execution_mode == 'serial'):
            t1 = time.time()
            for avatar_id in self.simulated_avatars_id:
                self.simulate_one_avatar(avatar_id)
            t2 = time.time()
            print(f"Time cost: {t2-t1}s")

        elif(self.execution_mode == 'parallel'):
            import asyncio
            from concurrent.futures import ThreadPoolExecutor
            loop = asyncio.get_event_loop()
            sim_workers = int(os.getenv("ARENA_MAX_WORKERS", "32"))
            sim_workers = max(1, min(sim_workers, len(self.simulated_avatars_id)))
            print(f"[Arena] simulation parallel workers={sim_workers}")
            executor = ThreadPoolExecutor(max_workers=sim_workers)
            tasks = []

            t1 = time.time()
            for avatar_id in self.simulated_avatars_id:
                tasks.append(self.async_simulate_one_avatar(avatar_id, loop, executor))
            loop.run_until_complete(asyncio.wait(tasks))
            t2 = time.time()
            print(f"Time cost: {t2-t1}s")

    async def async_simulate_one_avatar(self, avatar_id, loop, executor):
        """
        async
        excute the simulation for one avatar
        avatar_id: the id of the simulated avatar
        """
        start_time = time.time()
        time_local = time.localtime(start_time)
        l_start = time.strftime("%Y-%m-%d %H:%M:%S",time_local)
        with open(self.storage_base_path + "/system_log.txt", 'a') as f:
            f.write(f"Start: {l_start}. User {avatar_id} starts simulation.\n")

        avatar_ = self.avatars[avatar_id]
        avatar_.write_log(f"Is simulating avatar {avatar_id}")
        avatar_.exit_flag = False
        page_generator = self.page_generator(avatar_id)
        i = 0
        user_behavior_dict = {}
        user_interview_dict = {}
        while not avatar_.exit_flag:
            i += 1
            id_on_page = next(page_generator, []) # get the next page, a list of item ids
            if(len(id_on_page) == 0):
                break
            movies_on_page = [self.movie_detail.loc[idx] for idx in id_on_page] # movie_detail.csv
            if self.use_beauty_prompt:
                recommended_items = []
                for idx, item in enumerate(movies_on_page):
                    tags = [t for t in str(getattr(item, "genres", "")).split("|") if t]
                    tags_str = ", ".join(tags[:6]) if tags else "N/A"
                    recommended_items.append(
                        "<- ID: " + str(int(id_on_page[idx])) + " ->"
                        + " <- Title: " + str(item.title) + " ->"
                        + " <- Avg rating: " + str(round(float(item.rating), 2)) + " ->"
                        + " <- Tags: " + tags_str + " ->"
                        + " <- Summary: " + str(item.summary) + " ->" + "\n"
                    )
            else:
                recommended_items = ["<- " + item.title + " ->" 
                                # + " <- Genres: " + (',').join(list(item.genres.split('|'))) + " ->"
                                + " <- History ratings: " + str(round(item.rating,2)) + " ->" 
                                + " <- Summary: " + item.summary + " ->" + "\n"
                                for item in movies_on_page]
            
            if(self.add_advert):
                #store_path = op.join(f"storage/{self.dataset}/{self.modeltype}/{self.simulation_name}/adver_id", f"avatar{avatar_id}_{i}.txt")
                store_path = f"storage/{self.dataset}/{self.modeltype}/{self.simulation_name}/adver_id"
                if not os.path.exists(store_path):
                    os.makedirs(store_path)
                if not self.display_advert:
                    recommended_items[0], id_on_page, movies_on_page = self.display_only_adver_item(store_path, avatar_id, i, id_on_page, movies_on_page)
                else:
                    recommended_items[0], id_on_page, movies_on_page = self.display_item_with_adver(store_path, avatar_id, i, id_on_page, movies_on_page)


            recommended_items_str = ''.join(recommended_items)

            # Please write down the recommended information.
            avatar_.write_log(f"\n=============    Recommendation Page {i}    =============")
            for idx, movie in enumerate(movies_on_page):
                if(id_on_page[idx] in self.data.valid_user_list[avatar_id]):
                    avatar_.write_log(f"== (√) {movie.title} History ratings: {round(movie.rating,2)} Summary: {movie.summary}", "blue", attrs=["bold"])
                else:
                    avatar_.write_log(f"== {movie.title} History ratings: {round(movie.rating,2)} Summary: {movie.summary}")
            avatar_.write_log(f"=============          End Page {i}        =============\n")

            # As a translator, I will translate the Chinese sentence you sent me into English. I do not need to understand the meaning of the content to provide a response.
            avatar_.write_log(f"\n==============    Avatar {avatar_.avatar_id} Response {i}   =============")


            # @ most important Waiting for user response.
            response = await loop.run_in_executor(executor, avatar_.reaction_to_recommended_items, recommended_items_str, i)

            #==============================================
            parsed_response = self._parse_reaction_output(
                response,
                candidate_titles=[movie.title for movie in movies_on_page],
            )
            align_entries = parsed_response["align_entries"]
            rating_entries = parsed_response["rating_entries"]
            watched_movies = parsed_response["watch_titles"]

            if self.add_advert and align_entries and align_entries[0]["align"] == "yes":
                self.clicked_adverts += 1
            
            title_id_dict = dict(zip(self.movie_detail["title"], self.movie_detail["movie_id"]))
            def resolve_item_id(token):
                token = self._normalize_field(token)
                if not token:
                    return None
                if token.isdigit():
                    iid = int(token)
                    if 0 <= iid < len(self.movie_detail):
                        return iid
                    return None
                return title_id_dict.get(token)
            like_movies = [entry for entry in rating_entries if entry["rating"] == 5]
            align_movies = [entry for entry in align_entries if entry["align"] == "yes"]

            info_on_page = {}
            info_on_page['page'] = i
            info_on_page['ground_truth'] = [id_on_page[idx] for idx, movie in enumerate(movies_on_page) if id_on_page[idx] in self.data.valid_user_list[avatar_id]]
            info_on_page['recommended_id'] = id_on_page
            info_on_page['recommended'] = [self.movie_detail.loc[idx].title for idx in id_on_page]
            info_on_page['align_id'] = [iid for iid in (resolve_item_id(entry["title"]) for entry in align_movies) if iid is not None]
            info_on_page['like_id'] = [iid for iid in (resolve_item_id(entry["title"]) for entry in like_movies) if iid is not None]
            info_on_page['watch_id'] = [iid for iid in (resolve_item_id(title) for title in watched_movies) if iid is not None]
            info_on_page['watched'] = watched_movies
            info_on_page['rating_id'] = [resolve_item_id(entry["title"]) for entry in rating_entries]
            info_on_page['rating'] = [entry["rating"] for entry in rating_entries]
            info_on_page['feeling'] = [entry["feeling"] for entry in rating_entries]
            info_on_page['align_reason'] = [entry["reason"] for entry in align_entries]
            info_on_page['watch_reason'] = parsed_response["watch_reason"]
            user_behavior_dict[i] = info_on_page

            # @ Add new training data.
            # new_train = [id_on_page[idx] for idx, movie, reason in like_movies] # Add all liked item ids in the validation set to the training set.
            # tmp = [(idx, movie_title.strip(';'), feeling.strip(';')) for idx, (movie_title, rating, feeling) in enumerate(match1[:self.items_per_page])]
            new_train = info_on_page['align_id']
            self.new_train_dict[avatar_id].extend(new_train)

            # @ Record the average number of likes.
            self.n_likes[avatar_id].append(len(new_train))
            ratings_list = info_on_page['rating']
            average_rating = sum(ratings_list) / len(ratings_list) if ratings_list else 0
            # Add the average score of this page.
            self.ratings[avatar_id].append(average_rating)
            self.watch[avatar_id].extend(watched_movies)

            # @ Calculate the precision on this page and save it.
            ground_truth = [id_on_page[idx] for idx, movie in enumerate(movies_on_page) if id_on_page[idx] in self.data.valid_user_list[avatar_id]]
            # print(like_movies, ground_truth)
            perf = (len(set(new_train) & set(ground_truth)), len(new_train), len(ground_truth))
            self.perf_per_page[avatar_id].append(perf)
            #==============================================

            vars.global_finished_pages += 1

            # @ Force exit if the number of pages exceeds the maximum limit.
            if(i >= self.max_pages):
                avatar_.exit_flag = True
        
        interview_response = avatar_.response_to_question("Do you feel satisfied with the recommender system you have just interacted? Rate this recommender system from 1-10 and give explanation.\n Please use this respond format: RATING: [integer between 1 and 10]; REASON: [explanation]; In RATING part just give your rating and other reason and explanation should included in the REASON part.", remember=False)
        # Extract RAING and REASON using re.
        pattern_interview = re.compile(r'RATING:\s*(.*?)\s*REASON:\s*(.*?)')
        # pattern_interview = re.compile(r'RATING:\s*(.*?)\s*REASON:\s*(.*?)')
        #pattern = re.compile(r'MOVIE:\s*(.*?)\s*WATCH:\s*(.*?)\s*REASON:\s*(.*?)\s*RATING:\s*(.*?)\s*FEELING:(.*?)')
        matches_interview = re.findall(r'(?<=RATING:|REASON:).*', interview_response)
        user_interview_dict['interview'] = matches_interview
        self._safe_console_print(matches_interview)
        self.exit_page[avatar_id] = i
        self.finished_num += 1
        self.remaining_users.remove(avatar_id)
        remaining = ", ".join([str(u) for u in self.remaining_users])

        end_time = time.time()
        time_local = time.localtime(end_time)
        l_end = time.strftime("%Y-%m-%d %H:%M:%S",time_local)
        vars.global_finished_users += 1
        with open(self.storage_base_path + "/system_log.txt", 'a') as f:
            f.write(f"Start: {l_start} End: {l_end}. User {avatar_id} finished after {i} pages. [{self.finished_num} / {self.n_avatars}]. Total token cost: {round(self.avatars[avatar_id].memory.user_k_tokens, 2)}k. Taking {round(time.time() - start_time, 2)}s\n")
            f.write(f"Remaining users: {remaining}\n")

        # @ Save the behavior of each individual.
        behavior_path = self.storage_base_path+ "/behavior"
        if not os.path.exists(behavior_path):
            os.makedirs(behavior_path)
        with open(behavior_path + f"/{avatar_id}.pkl", 'wb') as f:
            pickle.dump(user_behavior_dict, f)

        interview_path = self.storage_base_path+ "/interview"
        if not os.path.exists(interview_path):
            os.makedirs(interview_path)
        with open(interview_path + f"/{avatar_id}.pkl", 'wb') as f:
            pickle.dump(user_interview_dict, f)

    def simulate_one_avatar(self, avatar_id):
        """
        excute the simulation for one avatar
        avatar_id: the id of the simulated avatar
        """
        # print("\nIs simulating avatar {}".format(avatar_id))
        avatar_ = self.avatars[avatar_id]
        avatar_.write_log(f"Is simulating avatar {avatar_id}")
        avatar_.exit_flag = False
        page_generator = self.page_generator(avatar_id)
        i = 0
        while not avatar_.exit_flag:
        # for i in range(2):
            i += 1
            id_on_page = next(page_generator, []) # get the next page, a list of item ids
            if(len(id_on_page) == 0):
                break

            movies_on_page = [self.movie_detail.loc[idx] for idx in id_on_page]
            if self.use_beauty_prompt:
                recommended_items = []
                for idx, item in enumerate(movies_on_page):
                    tags = [t for t in str(getattr(item, "genres", "")).split("|") if t]
                    tags_str = ", ".join(tags[:6]) if tags else "N/A"
                    recommended_items.append(
                        "<- ID: " + str(int(id_on_page[idx])) + " ->"
                        + " <- Title: " + str(item.title) + " ->"
                        + " <- Avg rating: " + str(round(float(item.rating), 2)) + " ->"
                        + " <- Tags: " + tags_str + " ->"
                        + " <- Summary: " + str(item.summary) + " ->" + "\n"
                    )
            else:
                recommended_items = ["<- " + item.title + " ->" 
                                + " <- History ratings: " + str(round(item.rating,2)) + " ->" 
                                + " <- Summary: " + item.summary + " ->" + "\n"
                                for item in movies_on_page]
            recommended_items_str = ''.join(recommended_items)
            avatar_.write_log("=============    Recommendation Page    =============")
            for idx, movie in enumerate(movies_on_page):
                if(id_on_page[idx] in self.data.valid_user_list[avatar_id]):
                    avatar_.write_log(f"== {movie} (√)", "blue", attrs=["bold"])
                else:
                    avatar_.write_log(f"== {movie}")
            avatar_.write_log("=============          End Page         =============")
            avatar_.write_log("")
            
            #@ most important
            response = avatar_.reaction_to_recommended_items(recommended_items_str, i)

            avatar_.write_log("")
            avatar_.write_log("=============    Avatar Response    =============")
            avatar_.write_log(response, color='yellow', attrs=None)
    
    def parse_response(self, response):
        #pattern = re.compile(r'MOVIE:\s*(.*?)\s*WATCH:\s*(.*?)\s*REASON:\s*(.*?)\s*FEELING:\s*(.*?)\s*RATING:\s*(\d)')
        pattern = re.compile(r'MOVIE:\s*(.*?)\s*WATCH:\s*(.*?)\s*REASON:\s*(.*?)\s*RATING:\s*(.*?)\s*FEELING:(.*?)')
        matches = re.findall(pattern, response)

        watched_movies, watched_movies_contain_id = [], []

        for idx, (movie_title, watch, reason, rating, feeling) in enumerate(matches):
            if(self.add_advert and idx == 0 and watch.strip(';') == 'yes'): # If the first one has an advertisement and the user clicked on it.
                self.clicked_adverts += 1
            if(watch.strip(';') == 'yes'):
                watched_movies.append(movie_title.strip(';'))
            self._safe_console_print((movie_title, watch, reason, rating, feeling))
        return response

    def display_only_adver_item(self, store_path, avatar_id, i, id_on_page, movies_on_page):
        store_path = op.join(store_path, f"avatar{avatar_id}_{i}.txt")
        try:
            with open(store_path, 'r') as f1:
                random_key = int(f1.read())
        except:
            try:
                store_path_minus_1 = op.join(store_path, f"avatar{avatar_id}_{i-1}.txt")
                with open(store_path_minus_1, 'r') as f2:
                    random_key = int(f2.read())
            except:
                store_path_minus_2 = op.join(store_path, f"avatar{avatar_id}_{i-2}.txt")
                with open(store_path_minus_2, 'r') as f3:
                    random_key = int(f3.read())
                    try:
                        store_path_minus_3 = op.join(store_path, f"avatar{avatar_id}_{i-3}.txt")
                        with open(store_path_minus_3, 'r') as f4:
                            random_key = int(f4.read())
                    except:
                            store_path_minus_4 = op.join(store_path, f"avatar{avatar_id}_{i-4}.txt")
                            with open(store_path_minus_4, 'r') as f5:
                                random_key = int(f5.read())


        self.total_adverts += 1
        id_on_page[0] = random_key
        movies_on_page[0] = self.movie_detail.loc[random_key]
        adver_information = self.advert[random_key]

        return ( "<- " + adver_information['title'] + " ->" 
                                + " <- History ratings:" + str(round(adver_information['rating'], 2)) + " ->"
                                + " <- Summary:" + adver_information['summary'] + " ->" + "\n"), id_on_page, movies_on_page

    def display_item_with_adver(self, store_path, avatar_id, i, id_on_page, movies_on_page):
        store_path = op.join(store_path, f"avatar{avatar_id}_{i}.txt")
        random_key = np.random.choice(list(self.advert.keys()))
        self.total_adverts += 1
        random_advert = self.advert[random_key]
        id_on_page[0] = random_key
        movies_on_page[0] = self.movie_detail.loc[random_key]
        advert_item_id = random_key

        with open(store_path, 'w') as f:
            f.write(f"{advert_item_id}")
        
        return ( self.advert_word 
                + "<- " + random_advert['title'] + " ->" 
                + "<- " + random_advert['review'] + " ->"
                + " <- History ratings:" + str(round(random_advert['rating'], 2)) + " ->" 
                + " <- Summary:" + random_advert['summary'] + " ->" + "\n"), id_on_page, movies_on_page

    # NOTE: This overrides the older serial simulate_one_avatar implementation above, which
    # did not respect --max_pages and did not update metrics/new_train/per-page logs.
    def simulate_one_avatar(self, avatar_id):
        """
        Execute the simulation for one avatar (serial mode).
        """
        start_time = time.time()
        l_start = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
        with open(self.storage_base_path + "/system_log.txt", "a") as f:
            f.write(f"Start: {l_start}. User {avatar_id} starts simulation.\n")

        avatar_ = self.avatars[avatar_id]
        avatar_.write_log(f"Is simulating avatar {avatar_id}")
        avatar_.exit_flag = False
        page_generator = self.page_generator(avatar_id)
        i = 0
        user_behavior_dict = {}
        user_interview_dict = {}

        while not avatar_.exit_flag:
            i += 1
            id_on_page = next(page_generator, [])
            if len(id_on_page) == 0:
                break

            movies_on_page = [self.movie_detail.loc[idx] for idx in id_on_page]

            if self.use_beauty_prompt:
                recommended_items = []
                for idx, item in enumerate(movies_on_page):
                    tags = [t for t in str(getattr(item, "genres", "")).split("|") if t]
                    tags_str = ", ".join(tags[:6]) if tags else "N/A"
                    recommended_items.append(
                        "<- ID: " + str(int(id_on_page[idx])) + " ->"
                        + " <- Title: " + str(item.title) + " ->"
                        + " <- Avg rating: " + str(round(float(item.rating), 2)) + " ->"
                        + " <- Tags: " + tags_str + " ->"
                        + " <- Summary: " + str(item.summary) + " ->\n"
                    )
            else:
                recommended_items = [
                    "<- " + item.title + " ->"
                    + " <- History ratings: " + str(round(item.rating, 2)) + " ->"
                    + " <- Summary: " + item.summary + " ->\n"
                    for item in movies_on_page
                ]

            if self.add_advert:
                store_path = f"storage/{self.dataset}/{self.modeltype}/{self.simulation_name}/adver_id"
                if not os.path.exists(store_path):
                    os.makedirs(store_path)
                if not self.display_advert:
                    recommended_items[0], id_on_page, movies_on_page = self.display_only_adver_item(
                        store_path, avatar_id, i, id_on_page, movies_on_page
                    )
                else:
                    recommended_items[0], id_on_page, movies_on_page = self.display_item_with_adver(
                        store_path, avatar_id, i, id_on_page, movies_on_page
                    )

            recommended_items_str = "".join(recommended_items)

            avatar_.write_log(f"\n=============    Recommendation Page {i}    =============")
            for idx, movie in enumerate(movies_on_page):
                prefix = "[GT] " if id_on_page[idx] in self.data.valid_user_list[avatar_id] else ""
                avatar_.write_log(
                    f"== {prefix}{movie.title} History ratings: {round(movie.rating,2)} Summary: {movie.summary}"
                )
            avatar_.write_log(f"=============          End Page {i}        =============\n")
            avatar_.write_log(f"\n==============    Avatar {avatar_.avatar_id} Response {i}   =============")

            response = avatar_.reaction_to_recommended_items(recommended_items_str, i)

            parsed_response = self._parse_reaction_output(
                response,
                candidate_titles=[movie.title for movie in movies_on_page],
            )
            align_entries = parsed_response["align_entries"]
            rating_entries = parsed_response["rating_entries"]
            watched_movies = parsed_response["watch_titles"]

            if self.add_advert and align_entries and align_entries[0]["align"] == "yes":
                self.clicked_adverts += 1

            title_id_dict = dict(zip(self.movie_detail["title"], self.movie_detail["movie_id"]))

            def resolve_item_id(token):
                token = self._normalize_field(token)
                if not token:
                    return None
                if token.isdigit():
                    iid = int(token)
                    if 0 <= iid < len(self.movie_detail):
                        return iid
                    return None
                return title_id_dict.get(token)

            like_movies = [entry for entry in rating_entries if entry["rating"] == 5]
            align_movies = [entry for entry in align_entries if entry["align"] == "yes"]

            info_on_page = {
                "page": i,
                "ground_truth": [
                    id_on_page[idx]
                    for idx, movie in enumerate(movies_on_page)
                    if id_on_page[idx] in self.data.valid_user_list[avatar_id]
                ],
                "recommended_id": id_on_page,
                "recommended": [self.movie_detail.loc[idx].title for idx in id_on_page],
                "align_id": [
                    iid
                    for iid in (resolve_item_id(entry["title"]) for entry in align_movies)
                    if iid is not None
                ],
                "like_id": [
                    iid
                    for iid in (resolve_item_id(entry["title"]) for entry in like_movies)
                    if iid is not None
                ],
                "watch_id": [
                    iid for iid in (resolve_item_id(title) for title in watched_movies) if iid is not None
                ],
                "watched": watched_movies,
                "rating_id": [resolve_item_id(entry["title"]) for entry in rating_entries],
                "rating": [entry["rating"] for entry in rating_entries],
                "feeling": [entry["feeling"] for entry in rating_entries],
                "align_reason": [entry["reason"] for entry in align_entries],
                "watch_reason": parsed_response["watch_reason"],
            }
            user_behavior_dict[i] = info_on_page

            new_train = info_on_page["align_id"]
            self.new_train_dict[avatar_id].extend(new_train)

            self.n_likes[avatar_id].append(len(new_train))
            ratings_list = info_on_page["rating"]
            average_rating = sum(ratings_list) / len(ratings_list) if ratings_list else 0
            self.ratings[avatar_id].append(average_rating)
            self.watch[avatar_id].extend(watched_movies)

            perf = (len(set(new_train) & set(info_on_page["ground_truth"])), len(new_train), len(info_on_page["ground_truth"]))
            self.perf_per_page[avatar_id].append(perf)

            vars.global_finished_pages += 1

            if i >= self.max_pages:
                avatar_.exit_flag = True

        interview_response = avatar_.response_to_question(
            "Do you feel satisfied with the recommender system you have just interacted? Rate this recommender system from 1-10 and give explanation.\n Please use this respond format: RATING: [integer between 1 and 10]; REASON: [explanation]; In RATING part just give your rating and other reason and explanation should included in the REASON part.",
            remember=False,
        )
        matches_interview = re.findall(r"(?<=RATING:|REASON:).*", interview_response)
        user_interview_dict["interview"] = matches_interview
        self._safe_console_print(matches_interview)

        self.exit_page[avatar_id] = i
        self.finished_num += 1
        if avatar_id in self.remaining_users:
            self.remaining_users.remove(avatar_id)
        remaining = ", ".join([str(u) for u in self.remaining_users])

        l_end = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        vars.global_finished_users += 1
        with open(self.storage_base_path + "/system_log.txt", "a") as f:
            f.write(
                f"Start: {l_start} End: {l_end}. User {avatar_id} finished after {i} pages. [{self.finished_num} / {self.n_avatars}]. Total token cost: {round(self.avatars[avatar_id].memory.user_k_tokens, 2)}k. Taking {round(time.time() - start_time, 2)}s\n"
            )
            f.write(f"Remaining users: {remaining}\n")

        behavior_path = self.storage_base_path + "/behavior"
        if not os.path.exists(behavior_path):
            os.makedirs(behavior_path)
        with open(behavior_path + f"/{avatar_id}.pkl", "wb") as f:
            pickle.dump(user_behavior_dict, f)

        interview_path = self.storage_base_path + "/interview"
        if not os.path.exists(interview_path):
            os.makedirs(interview_path)
        with open(interview_path + f"/{avatar_id}.pkl", "wb") as f:
            pickle.dump(user_interview_dict, f)

    def save_results(self):
        """
        save the results of the simulation
        """
        # if(self.n_avatars == self.data.n_users):
        def save_user_dict_to_txt(user_dict, base_path, filename):
            with open(base_path + filename, 'w') as f:
                for u, v in user_dict.items():
                    f.write(str(int(u)))
                    for i in v:
                        f.write(' ' + str(int(i)))
                    f.write('\n')

        # save_path = f"datasets/{self.dataset}_{self.modeltype}/cf_data/"
        save_path = f"storage/{self.dataset}/{self.modeltype}/{self.simulation_name}/"
        save_user_dict_to_txt(self.new_train_dict, save_path, 'train.txt')

        # @ Save overall evaluation indicators.
        # Average number of clicks per user
        cprint("Number of likes", color='green', attrs=['bold'])
        cprint(self.n_likes, color='green', attrs=['bold'])
        average_n_likes = {avatar_id:np.mean(n_likes) for avatar_id, n_likes in self.n_likes.items()}
        cprint(average_n_likes, color='green', attrs=['bold'])

        overall_n_likes = np.mean(list(average_n_likes.values()))
        cprint(f"\nOverall number of likes: {overall_n_likes}", color='green', attrs=['bold'])

        # Average satisfaction
        cprint("\nRatings", color='green', attrs=['bold'])
        cprint(self.ratings, color='green', attrs=['bold'])
        average_ratings = {avatar_id:np.mean(ratings) for avatar_id, ratings in self.ratings.items()}
        cprint(average_ratings, color='green', attrs=['bold'])

        # @ Save average click-through rate
        # Use *real* exposure as denominator: how many items were actually shown
        # before the user exited, rather than the theoretical max_pages*items_per_page.
        average_click_rate = {
            avatar_id: len(movies) / max(int(self.exit_page.get(avatar_id, 0)) * self.items_per_page, 1)
            for avatar_id, movies in self.watch.items()
        }
        cprint(f"\nAverage click rate: {average_click_rate}", color='green', attrs=['bold'])
        overall_click_rate = np.mean(list(average_click_rate.values()))
        cprint(f"\nOverall satisfaction: {overall_click_rate}", color='green', attrs=['bold']) # Average click-through rate

        # overall_click_rate = np.mean(list(average_ratings.values()))
        # cprint(f"\nOverall satisfaction: {overall_click_rate}", color='green', attrs=['bold'])

        # Average exit page
        mean_exit_page = np.mean(list(self.exit_page.values()))
        cprint("\nExit pages", color='green', attrs=['bold'])
        cprint(self.exit_page, color='green', attrs=['bold'])
        cprint(f"Average exit page: {mean_exit_page}", color='green', attrs=['bold'])

        # Average precision and recall
        cprint("\nPrecision and recall", color='green', attrs=['bold'])
        cprint(self.perf_per_page, color="green", attrs=['bold'])
        total_perf = {avatar_id:[sum([i for i, j, k in perf_per_page]), sum([j for i, j, k in perf_per_page]), sum([k for i, j, k in perf_per_page])] for avatar_id, perf_per_page in self.perf_per_page.items()}
        total_recall_precision = {avatar_id:(perf[0]/max(perf[1], 1), perf[0]/max(perf[2], 1)) for avatar_id, perf in total_perf.items()}
        cprint(total_perf, color="green", attrs=['bold'])
        cprint(total_recall_precision, color="green", attrs=['bold'])
        average_precision = np.mean([metrics[0] for avatar_id, metrics in total_recall_precision.items()])
        average_recall = np.mean([metrics[1] for avatar_id, metrics in total_recall_precision.items()])
        cprint(f"Precision: {average_precision}  Recall: {average_recall}", color="green", attrs=['bold'])
        # metrics_path = self.storage_base_path + "/metrics.txt"
        total_k_tokens = sum([self.avatars[i].memory.user_k_tokens for i in range(self.n_avatars)])

        # Effective advertising rate
        if(self.add_advert):
            cprint("\nAdvert", color='green', attrs=['bold'])
            cprint(f"Total advert: {self.total_adverts}", color='green', attrs=['bold'])
            cprint(f"Clicked advert: {self.clicked_adverts}", color='green', attrs=['bold'])
            cprint(f"Advert click rate: {self.clicked_adverts/self.total_adverts}", color='green', attrs=['bold'])

        end_time = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))
        with open(self.storage_base_path + "/metrics.txt", 'w') as f:
            f.write(f"Finished time: {end_time}\n")
            f.write(f"Total simulation time: {round(time.time() - self.start_time, 2)}s\n")
            f.write(f"n_avatars: {self.n_avatars}\n")
            f.write(f"Seed: {getattr(self.args, 'seed', '')}\n")
            f.write(f"LLM model: {getattr(self.args, 'llm_model', '')}\n")
            f.write(f"LLM API style: {getattr(self.args, 'llm_api_style', '')}\n")
            f.write(f"LLM temperature: {getattr(self.args, 'llm_temperature', '')}\n")
            f.write(f"Beauty prompt mode: {getattr(self.args, 'beauty_prompt_mode', '')}\n")
            f.write(f"Model path: {getattr(self.args, 'model_path', '')}\n")
            f.write(f"Hazard plan enabled: {bool(getattr(self.args, 'enable_hazard_plan', False))}\n")
            f.write(f"Average recall: {average_recall}\n")
            f.write(f"Average presion: {average_precision}\n")
            f.write(f"Total k tokens: {round(total_k_tokens, 2)}k tokens\n")
            f.write(f"Total cost: {round(total_k_tokens*0.0018, 2)} \n")
            # f.write(f"Average precision: {}")
            f.write(f"Maximum exit page: {self.max_pages}\n")
            f.write(f"Overall click rate: {overall_click_rate}\n")
            f.write(f"Average number of likes: {overall_n_likes}\n")
            f.write(f"Average exit page: {mean_exit_page}\n")
            if(self.add_advert):
                f.write(f"Total advert: {self.total_adverts}\n")
                f.write(f"Clicked advert: {self.clicked_adverts}\n")
                f.write(f"Advert click rate: {self.clicked_adverts/self.total_adverts}\n")
