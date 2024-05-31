from rl4lms.envs.text_generation.observation import Observation
from rl4lms.envs.text_generation.reward import RewardFunction
from rl4lms.envs.text_generation.metric import BaseMetric, RougeMetric, MSEMetric
from typing import Dict, Any, List
from transformers import AutoTokenizer
from transformers import PreTrainedModel
from myutil import get_generations_gpt3, ForkedPdb, levenshtein
from numpy import mean
import json
import random
import torch
import os, re, string
import ipdb
import csv
import numpy as np
from cache import AspectCacheKey, AspectCacheValue, AspectCache

CALLS = 0
ATTEMPTS = 0
HITS = 0


def build_tokenizer(tokenizer_config: Dict[str, Any]):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_config["model_name"])
    if tokenizer.pad_token is None and tokenizer_config.get(
        "pad_token_as_eos_token", True
    ):
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = tokenizer_config.get("padding_side", "left")
    tokenizer.truncation_side = tokenizer_config.get("truncation_side", "left")
    return tokenizer


class EditMatchMetric(BaseMetric):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.downstream_metric_name = kwargs["downstream_metric_name"]
        self.downstream_metric = metric_map[self.downstream_metric_name]
        self.prompt = kwargs["prompt_path"]
        self.separator = kwargs["separator"]
        self.openai_api_key = kwargs["openai_key"]
        self.model_name = kwargs["gpt3_model_name"]
        self.cache_path = kwargs["cache_path"]
        self.save_path = kwargs["save_path"]
        self.append_feedback_to_q = kwargs.get("append_feedback_to_q", False)
        self.lambda_rouge_input = kwargs.get("lambda_rouge_input", 0.3)
        self.aspect_cache = AspectCache(cache_file='cache.jsonl', save_every=5)



        assert self.downstream_metric_name in [
            "rouge_combined",
            "rouge_combined_diff",
            "rouge_combined_plus_rouge_input",
            "rougeC_diff_rouge_input",
            "loose_exact_match",
            "inverse_levenshtein",
            "inverse_levenshtein_diff",
            "inverse_levenshtein_diff_exact_match",
            "mse"
            ]

        # Load prompt.
        # print("-----", self.prompt)
        with open(self.prompt, "r") as f:
            self.prompt = f.read()

        # Check key is valid.
        if self.model_name != "code-davinci-002":
            raise ValueError("You will be charged by OpenAI for this run.")

        # Load cache from cache_path.
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "r") as f:
                self.GPT3_CACHE = json.load(f)

    def remove_prefix(self, text: str, prefixes: List[str]=["Generate schema for evaluating morality: ", "generate schema for evaluating morality: ", "Generate a schema to evaluate morality:"]):
        for prefix in prefixes:
            if text.startswith(prefix):
                return text[len(prefix):]
        return text

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
        epoch: int = None,
    ):
        global CALLS, ATTEMPTS, HITS

        # Strip off task prefix
        inputs = [self.remove_prefix(prompt) for prompt in prompt_texts]
        if self.append_feedback_to_q:
            # Prepend prompt.
            input_wfeed = []
            cached_wfeed = []
            for input_text, feedback_pred in zip(inputs, generated_texts):
                # Get text between "Question: " and "\n\nAnswer:"
                # it = input_text
                # it = it.replace("\n", " ")
                # question = re.search("Question: (.*)Answer:", it).group(1).strip()

                aspects = [aspect.strip() for aspect in feedback_pred.split(',')]
                
                uncached_aspects = {}
                cached_aspects = {}
                for i, aspect in enumerate(aspects):
                    cache_key = AspectCacheKey(scenario=input_text, aspect=aspect)
                    cache_val = self.aspect_cache.get(cache_key)
                    if cache_val:
                        cached_aspects[i] = (aspect, cache_val.response)
                    else:
                        uncached_aspects[i] = (aspect, None)
                feedback_pred = ', '.join(map(lambda x: x[0], list(uncached_aspects.values())))

                cached_wfeed.append(cached_aspects)
                if feedback_pred:
                    input_wfeed.append(
                        (
                            self.prompt
                            + self.separator
                            + "Scenario: "
                            + input_text
                            + "\nSchema: \n"
                            + feedback_pred
                            + "\n"
                        )
                    )
                else:
                    input_wfeed.append(None)
        else:
            # Prepend prompt.
            input_wfeed = [
                (
                    self.prompt
                    + self.separator
                    + input_text
                    + "\nFeedback: "
                    + feedback_pred
                    + "\nEdit:"
                )
                for input_text, feedback_pred in zip(inputs, generated_texts)
            ]
        
        # print("\n\nThis is input feed", input_wfeed)
        # print("-----------------\n\n")

        # if self.cache_path != "":
            
        #     try:
        #         self.GPT3_CACHE
        #     except:
        #         # If GPT3_CACHE is empty, load it from cache_path.
        #         if os.path.exists(self.cache_path):
        #             with open(self.cache_path, "r") as f:
        #                 self.GPT3_CACHE = json.load(f)
        #         else:
        #             self.GPT3_CACHE = {}

        #     # Check if we have cached results.
        #     cache_queries = [el for el in input_wfeed]

        #     cached_results = []
        #     uncached_inputs = []
        #     for i, input in enumerate(cache_queries):
        #         ATTEMPTS += 1
        #         if input in self.GPT3_CACHE:
        #             HITS += 1
        #             cached_results.append((i, self.GPT3_CACHE[input]))
        #         else:
        #             uncached_inputs.append((i, input_wfeed[i]))
        #     input_wfeed = [x[1] for x in uncached_inputs]

        # Query GPT-3

        gpt_inputs = []

        for inp in input_wfeed:
            if inp:
                # then we have to me a call
                gpt_inputs.append((inp, True))
            else:
                # then it has been cached
                gpt_inputs.append((cached_wfeed, False))

        gpt_call_1 = get_generations_gpt3(
            ls=gpt_inputs,
            model_name=self.model_name,
            clean_tok=True,
            stop=[self.separator, "Edit:", "Feedback:", "Question:"],
            temperature=0.0,
            batch_size=1,
            max_length=150,
            penalty=0.0,
            n=1,
            keyfile=self.openai_api_key,
            generate_json=True
        )
        
        # merge cached and uncaches results
        merged_results = []
        for scenario, uncached_scores, cached_scores in zip(inputs, gpt_call_1, cached_wfeed):

            # 
            merged_aspect_scores = {}
            num_aspects = len(uncached_scores) + len(cached_scores)

            uncached_aspect_counter = 0
            uncached_aspect_keys = list(uncached_scores.keys())
            for i in range(num_aspects):
                cached_res = cached_scores.get(i, None)
                if cached_res:
                    merged_aspect_scores[cached_res[0]] = cached_res[1]

                else:
                    uncached_aspect = uncached_aspect_keys[uncached_aspect_counter]
                    merged_aspect_scores[uncached_aspect] = uncached_scores[uncached_aspect]
                    uncached_aspect_counter += 1
                    cache_key = AspectCacheKey(scenario=scenario, aspect=uncached_aspect)
                    cache_val = AspectCacheValue(response=uncached_scores[uncached_aspect])
                    self.aspect_cache.set(cache_key, cache_val)
            merged_results.append(merged_aspect_scores)

            

        message = """
            You are given a set of aspects and a score for each aspect that tells how the aspect was evaluated for some situation. 
             
            Score Interpretation:

            Positive Scores [1 to 4]: The aspect was positively evaluated (e.g., 1 = low positive evaluation, 4 = high positive evaluation).
            Negative Scores [-4 to -1]: The aspect was negatively evaluated (e.g., -1 = low negative evaluation, -4 = high negative evaluation).
            Neutral Score [0]: The scenario is neutral for this aspect.
            
            Aspects:

            {{virtue set}}

            Task:

            Carefully analyze the individual scores of each aspect and choose the final morality score
            based on the aspects from the set [-4, -3, -2, -1, 0, 1, 2, 3, 4] with -4 being highly immoral and 4 being highly moral.

            Note: The morality score should be strictly an integer in the range -4 to 4.

            Response Format:
            Provide the score in strict JSON format:
            
            {
                "morality_score": "... integer score in the range -4 to 4"
            }
        """

        all_messages = []

        for obj in merged_results:
            arr = []
            for k, v in obj.items():
                arr.append(f'{k}: {v}')
            
            final_str = "\n".join(arr)
            new_message = message.replace("{{virtue set}}", final_str)

            all_messages.append((new_message, True))

        # here, we need to merge


        final_preds = get_generations_gpt3(
            ls=all_messages,
            model_name=self.model_name,
            clean_tok=True,
            stop=[self.separator, "Edit:", "Feedback:", "Question:"],
            temperature=0.0,
            batch_size=1,
            max_length=150,
            penalty=0.0,
            n=1,
            keyfile=self.openai_api_key,
            generate_json=True,
            final_morality=True
        )

        print("\n\nThis is the final GPT output, ", final_preds)

        edit_pred = []
        for obj in final_preds:
            try:
                edit_pred.append(str(obj["morality_score"]))
            except:
                edit_pred.append(str(0))

        print("\n\n This is the edit pred", edit_pred)

        # if self.cache_path != "":
        #     # Update cache.
        #     uncached_queries = [cache_queries[i] for i, _ in uncached_inputs]
        #     self.GPT3_CACHE.update(dict(zip(uncached_queries, edit_pred)))

        #     if CALLS % 100 == 0:
        #         print("Size: ", len(self.GPT3_CACHE), "Attempts: ", ATTEMPTS, "Hits: ", HITS, "Ratio: ", HITS / ATTEMPTS)
        #         print("Saving cache to", self.cache_path)
        #         with open(self.cache_path, "w") as f:
        #             json.dump(self.GPT3_CACHE, f)
        #     CALLS += 1

        #     edit_pred = iter(edit_pred)
        #     uncached_results = [(i, next(edit_pred)) for i, _ in uncached_inputs]

        #     # Combine cached and uncached results.
        #     results = cached_results + uncached_results

        #     # Sort results by index.
        #     results.sort(key=lambda x: x[0])
        #     edit_pred = [v for _, v in results]

        # If len(prompt_texts) = 1, then print.
        if len(edit_pred) <= 2:
            # print("!!! prompt_text:\t", prompt_texts)
            # print("!!! generated_text:\t", generated_texts)
            # print("!!! reference_texts:\t", reference_texts)
            # print("!!! edit_pred:\t", edit_pred)
            with open("data/ppo_generation.csv", mode="a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([prompt_texts, generated_texts, reference_texts, edit_pred])

        if self.downstream_metric_name == "rouge_combined_plus_rouge_input":
            # Strip off the part that starts with "Question:", if any.
            inputs = [re.sub("Question:.*", "", input) for input in inputs]
            scores = self.downstream_metric(edit_pred, reference_texts, generated_texts, inputs, self.lambda_rouge_input)

        elif self.downstream_metric_name in ["rouge_combined_diff", "inverse_levenshtein_diff", "inverse_levenshtein_diff_exact_match"]:
            # TODO: These if/else statements are due to data formatting differences. Fixme.
            if "Answer:" in inputs[0]:
                # Retrieve the part that is after "Answer:"
                init_pred = [re.search("Answer:(.*)", input).group(1).strip() for input in inputs]
            elif "Steps:" in inputs[0]:
                init_pred = [re.search("Steps:(.*)", input).group(1).strip() for input in inputs]
            elif "|||" in inputs[0]:
                # Get the part before "Feedback:"
                init_pred = [re.search("\|\|\|(.*)", input).group(1).strip() for input in inputs]
            else:
                raise ValueError("Unknown input format, cannot extract initial prediction.")
            scores = self.downstream_metric(edit_pred, reference_texts, init_pred)

        elif self.downstream_metric_name == "rougeC_diff_rouge_input":
            # TODO: These if/else statements are due to data formatting differences. Fixme.
            if "Answer:" in inputs[0]:
                # Retrieve the part that is after "Answer:"
                init_pred = [re.search("Answer:(.*)", input).group(1).strip() for input in inputs]
            elif "Steps:" in inputs[0]:
                init_pred = [re.search("Steps:(.*)", input).group(1).strip() for input in inputs]
            elif "|||" in inputs[0]:
                # Get the part before "Feedback:"
                init_pred = [re.search("\|\|\|(.*)", input).group(1).strip() for input in inputs]
            else:
                raise ValueError("Unknown input format, cannot extract initial prediction.")
            inputs = [re.sub("Question:.*", "", input) for input in inputs]
            scores = self.downstream_metric(
                pred=edit_pred,
                ref=reference_texts,
                feedback=generated_texts,
                inputs=inputs,
                init_pred=init_pred,
                lambda_rouge_input=self.lambda_rouge_input,
            )

        else:
            scores = self.downstream_metric(edit_pred, reference_texts)


        # Save edit_pred to save_path using split_name and epoch.
        if self.save_path != "" and split_name in ["test", "val"]:
            save_path = os.path.join(
                self.save_path, f"{split_name}_editmatch_{epoch}.json"
            )
            with open(save_path, "w") as f:
                json.dump(edit_pred, f)


        metric_dict = {}
        for k, score in scores.items():
            metric_dict.update({f"custom_metrics/editmatch_{k}": (None, score)})
        return metric_dict


class EditMatch(RewardFunction):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.tokenizer_config = kwargs["tokenizer"]
        self.tokenizer = build_tokenizer(self.tokenizer_config)
        self.metric = EditMatchMetric(**kwargs["metric"])

    def __call__(
        self,
        prev_observation: Observation,
        action: int,
        current_observation: Observation,
        done: bool,
        meta_info: Dict[str, Any] = None,
    ) -> float:

        if done:
            # 1. goal, steps, EOS, feedback_pred = Decode current_observation.input_encoded_pt
            # 2. edit_pred = query smallf (GPT-3)
            # 3. edit_gold = target_or_reference_texts
            # 4. reward = metric(edit_pred, edit_gold)

            state = current_observation.input_encoded_pt
            # print("STATEEETTEATET:", state)
            input_wfeed = self.tokenizer.decode(state[0], skip_special_tokens=True)

            # Get prompt and feedback separately.
            prompt_or_input_text = prev_observation.prompt_or_input_text
            edit_gold = current_observation.target_or_reference_texts

            prompt_texts = prompt_or_input_text.split('###')
            # print("This is edit_gold: ", edit_gold)
            reference_texts = [float(a) for a in edit_gold.split('###')]

            feedback_pred = input_wfeed.lstrip(prompt_or_input_text)
            # print("This is the input prompt or input text: ", prompt_or_input_text)
            # print("And this is the input wfeed: ", input_wfeed)
            # print("And this is feedback that is schema generateion: ", feedback_pred)
            prompt_or_input_text = prompt_or_input_text.lstrip("Critique: ")
            metric_dict = self.metric.compute(
                prompt_texts=prompt_texts,
                generated_texts=[feedback_pred] * len(prompt_texts),
                reference_texts=reference_texts,
            )
            reward = metric_dict[
                f"custom_metrics/editmatch_{self.metric.downstream_metric_name}"
            ][-1]
            return reward

        return 0


def mse_metric(pred: List[str], ref: List[str]) -> float:
    # pred = [float(json.loads(i)["morality_score"]) for i in pred]
    pred = np.array([float(i) for i in pred])
    ref = np.array([float(i) for i in ref])
    res = MSEMetric().compute(torch.tensor(pred), torch.tensor(ref))
    sign_reward = 16 * (pred * ref > 0)
    mse_penalty = res['semantic/mse'][-1]

    reward = sign_reward - mse_penalty.numpy()
    print('reward obtained', reward)
    return {"mse": reward.mean()}


def rouge1_metric(pred: List[str], ref: List[List[str]]):
    res = RougeMetric().compute(
        prompt_texts=[], generated_texts=pred, reference_texts=ref
    )
    return res["lexical/rouge_rouge1"][-1]


def rouge_combined(pred: List[str], ref: List[List[str]]):

    rouge_keys = ["rouge1", "rouge2", "rougeL"]
    res = RougeMetric(use_single_ref=False).compute(
        prompt_texts=[], generated_texts=pred, reference_texts=ref
    )
    rouge_scores = [res["lexical/rouge_" + k][-1] for k in rouge_keys]
    scores = dict(zip(rouge_keys, rouge_scores))
    scores.update({"rouge_combined": mean(rouge_scores)})
    return scores


def rouge_combined_diff(pred: List[str], ref: List[List[str]], init_pred: List[str]):
    scores = rouge_combined(pred, ref)
    scores_init = rouge_combined(init_pred, ref)
    scores_diff = {
        "rouge_combined_diff": scores["rouge_combined"] - scores_init["rouge_combined"],
        "rouge_combined_init": scores_init["rouge_combined"],
    }
    scores.update(scores_diff)
    return scores


def rouge_combined_plus_rouge_input(
        pred: List[str],
        ref: List[List[str]],
        feedback: List[str],
        inputs: List[str],
        lambda_rouge_input: float = 0.3,
    ):
    scores = rouge_combined(pred, ref)
    score_feedback = rouge1_metric(feedback, [[inp] for inp in inputs])
    combined_score = scores['rouge_combined'] + lambda_rouge_input * score_feedback
    scores.update({"rouge_combined_plus_rouge_input": combined_score})
    scores.update({"rouge1_input_feedback": score_feedback})

    return scores

def rougeC_diff_rouge_input(
        pred: List[str],
        ref: List[List[str]],
        feedback: List[str],
        inputs: List[str],
        init_pred: List[str],
        lambda_rouge_input: float,
    ):
    scores = rouge_combined(pred, ref)
    scores_init = rouge_combined(init_pred, ref)
    scores_diff = {
        "rouge_combined_diff": scores["rouge_combined"] - scores_init["rouge_combined"],
        "rouge_combined_init": scores_init["rouge_combined"],
    }
    scores.update(scores_diff)
    score_feedback = rouge1_metric(feedback, [[inp] for inp in inputs])
    combined_score = scores['rouge_combined_diff'] + lambda_rouge_input * score_feedback
    scores.update({"rougeC_diff_rouge_input": combined_score})
    scores.update({"rouge1_input_feedback": score_feedback})

    return scores


def loose_exact_match(pred: List[str], ref: List[List[str]], reduced=True) -> Dict[str, float]:
    """
    Checks if each element in pred matches at least one element in ref.
    Match should be case insensitive and ignore punctuation.
    """
    pred = [p.lower().translate(str.maketrans('', '', string.punctuation)) for p in pred]
    ref = [[r.lower().translate(str.maketrans('', '', string.punctuation)) for r in rs] for rs in ref]
    res = [any([p == r for r in rs]) for p, rs in zip(pred, ref)]
    if not reduced:
        return {"loose_exact_match": res}
    res = sum(res) / len(res)
    return {"loose_exact_match": res}


def inverse_levenshtein(pred: List[str], ref: List[List[str]], reduced=True) -> Dict[str, float]:
    """
    Computes the inverse of the levenshtein distance between pred and ref.
    The operation should be case insensitive and ignore punctuation.
    """
    pred = [p.lower().translate(str.maketrans('', '', string.punctuation)) for p in pred]
    ref = [[r.lower().translate(str.maketrans('', '', string.punctuation)) for r in rs] for rs in ref]
    res = [1 - levenshtein(p, rs[0]) / max(len(p), len(rs[0])) for p, rs in zip(pred, ref)]
    lem = loose_exact_match(pred, ref, reduced=reduced)["loose_exact_match"]
    if not reduced:
        return {"inverse_levenshtein": res, "loose_exact_match": lem}
    res = sum(res) / len(res)
    return {"inverse_levenshtein": res, "loose_exact_match": lem}

def inverse_levenshtein_diff(pred: List[str], ref: List[List[str]], init_pred: List[str]) -> Dict[str, float]:
    scores = inverse_levenshtein(pred, ref)
    scores_init = inverse_levenshtein(init_pred, ref)
    scores_diff = {
        "inverse_levenshtein_diff": scores["inverse_levenshtein"] - scores_init["inverse_levenshtein"],
        "inverse_levenshtein_init": scores_init["inverse_levenshtein"],
    }
    scores.update(scores_diff)
    scores.update(loose_exact_match(pred, ref))
    return scores


def inverse_levenshtein_diff_exact_match(pred: List[str], ref: List[List[str]], init_pred: List[str]) -> Dict[str, float]:
    scores = inverse_levenshtein(pred, ref, reduced=False)
    scores_init = inverse_levenshtein(init_pred, ref, reduced=False)
    lem = scores["loose_exact_match"]
    lem_init = scores_init["loose_exact_match"]
    final_score = []
    for i in range(len(scores["inverse_levenshtein"])):
        diff_score = scores["inverse_levenshtein"][i] - scores_init["inverse_levenshtein"][i]
        if diff_score >= 0:
            final_score.append(max(diff_score, lem[i]))
        else:
            final_score.append(diff_score)
    scores_diff = {
        "inverse_levenshtein_diff_exact_match": mean(final_score),
        "inverse_levenshtein_init": mean(scores_init["inverse_levenshtein"]),
        "inverse_levenshtein": mean(scores["inverse_levenshtein"]),
        "loose_exact_match": mean(lem),
        "loose_exact_match_init": mean(lem_init),
    }
    scores.update(scores_diff)
    print("!!! scores_diff:\t", scores)
    return scores


def custom_metric_scripting_func(pred: str, gold: str):
    """
    Args:
        pred: a string, should be in functional format e.g, [INSERT] node1 [AFTER] node2 [END]
        gold: a string, should be in natural language format e.g, Insert node1 after node2
    """
    score = 0.2
    pred = pred.replace("'", "")
    gold = gold.replace("'", "")

    try:
        if "[INSERT]" in pred and "[INSERT]" in gold:

            if "[AFTER]" in pred and "[AFTER]" in gold:

                node_insert = (
                    re.search("\[INSERT\](.*)\[AFTER\]", pred).group(1).strip()
                )
                node_after = re.search("\[AFTER\](.*)\[END\]", pred).group(1).strip()
                node_insert_g = (
                    re.search("\[INSERT\](.*)\[AFTER\]", gold).group(1).strip()
                )
                node_after_g = re.search("\[AFTER\](.*)\[END\]", gold).group(1).strip()

                if node_insert == node_insert_g:
                    score += 0.4
                if node_after == node_after_g:
                    score += 0.4

            elif "[BEFORE]" in pred and "[BEFORE]" in gold:

                node_insert = (
                    re.search("\[INSERT\](.*)\[BEFORE\]", pred).group(1).strip()
                )
                node_before = re.search("\[BEFORE\](.*)\[END\]", pred).group(1).strip()
                node_insert_g = (
                    re.search("\[INSERT\](.*)\[BEFORE\]", gold).group(1).strip()
                )
                node_before_g = (
                    re.search("\[BEFORE\](.*)\[END\]", gold).group(1).strip()
                )

                if node_insert == node_insert_g:
                    score += 0.4
                if node_before == node_before_g:
                    score += 0.4

            else:
                return score

        elif "[REMOVE]" in pred and "[REMOVE]" in gold:

            node_remove = re.search("\[REMOVE\](.*)\[END\]", pred).group(1).strip()
            node_remove_g = re.search("\[REMOVE](.*)\[END\]", gold).group(1).strip()

            if node_remove == node_remove_g:
                score += 0.8

        elif "[REORDER]" in pred and "[REORDER]" in gold:

            node_reorder1 = re.search("\[REORDER\](.*)\[AND\]", pred).group(1).strip()
            node_reorder2 = re.search("\[AND\](.*)\[END\]", pred).group(1).strip()

            node_reorder1_g = re.search("\[REORDER\](.*)\[AND\]", gold).group(1).strip()
            node_reorder2_g = re.search("\[AND\](.*)\[END\]", gold).group(1).strip()

            ss = set([node_reorder1, node_reorder1_g, node_reorder2, node_reorder2_g])
            if len(ss) == 2:
                score += 0.8
            if len(ss) == 3:
                score += 0.4

        else:
            return 0.0
    except AttributeError:
        return score
    return score


def exact_match_scripting(pred: str, gold: List[str]):
    score_list = []
    for ref in gold:
        score_list.append(custom_metric_scripting_func(pred, ref))

    score = max(score_list)

    return (score == 1.0) * 1.0, score


metric_map = {
    "custom": custom_metric_scripting_func,
    "rouge1": rouge1_metric,
    "rouge_combined": rouge_combined,
    "rouge_combined_diff": rouge_combined_diff,
    "rouge_combined_plus_rouge_input": rouge_combined_plus_rouge_input,
    "rougeC_diff_rouge_input": rougeC_diff_rouge_input,
    "loose_exact_match": loose_exact_match,
    "inverse_levenshtein": inverse_levenshtein,
    "inverse_levenshtein_diff": inverse_levenshtein_diff,
    "inverse_levenshtein_diff_exact_match": inverse_levenshtein_diff_exact_match,
    "mse": mse_metric
}


# TODO: can we do batched?
# TODO num of environs = 10
# (can this be a list to track
# rouge on feedback_pred vs feedback_gold and edit_pred vs edit_gold?)
if __name__ == "__main__":
    kwargs = {"tokenizer": "t5-base"}
    metric = EditMatch(**kwargs)

    args = {
        "downstream_metric_name": "mse",
        "prompt_path": "data/interscript/prompts_edit_numeric.txt",
        "separator": "\n\n---\n\n",
        "openai_key": "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",
        "gpt3_model_name": "code-davinci-002",
        "cache_path": "data/interscript/cache.json",
        # "cache_path": "data/interscript/cache_prompts_edit_functional_test.json",
    }

    metric = EditMatchMetric(**args)
    metric_dict = metric.compute(
        prompt_texts=[
            "Critique: Goal: plug in nightlight Steps: 1. find pillows and blankets 2. walk to nightlight 3. push button light on",
            "Critique: Goal: bring baby home Steps: 1. take baby 2. drop baby",
        ],
        generated_texts=["Should plug in the light", "you should drive home"],
        reference_texts=[
            ["[REMOVE] nightlight [END]"],
            ["[INSERT] drive home [AFTER] take baby [END]"],
        ],
    )
    print(metric_dict)
    print(CALLS)
