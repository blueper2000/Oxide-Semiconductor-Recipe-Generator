import os
from litellm import batch_completion, completion
import openai
import fire
from datasets import load_dataset, load_from_disk, concatenate_datasets
from pprint import pprint
import jsonlines
from tqdm import tqdm
from . import openai_utils
import numpy as np


class RecipePredictor:
        

    def __init__(self, 
                 model="gpt-4o-mini",
                 batch_size=1,
                 max_tokens=4096,
                 max_completion_tokens=16384,
                 temperature=0.0,
                 api_key=None,
                 prompt_filename: str="prompts/prediction.txt",
                 ):
        self.model = model
        self.prompt_filename = prompt_filename
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = openai.OpenAI(api_key=api_key)
        self.prediction_prompt = open(prompt_filename).read()
        self.job_description = f"material prediction job w/ {model}"
        self.max_completion_tokens = max_completion_tokens  

    def _get_model_config(self, model_name):
        """모델별 설정을 반환하는 통합 함수"""
        if model_name.startswith("gpt-5") or model_name.startswith("o4"):
            # GPT-5 모델들
            if model_name.startswith("gpt-5"):
                base_model = "gpt-5" if model_name != "gpt-5-mini" else "gpt-5-mini"
            else:  # o4 모델들
                base_model = "o4-mini"
            
            reasoning_effort = None
            if "high" in model_name:
                reasoning_effort = "high"
            elif "low" in model_name:
                reasoning_effort = "low"
            
            return base_model, reasoning_effort, True  # max_completion_tokens 사용
            
        elif model_name.startswith("o1") or model_name.startswith("o3"):
            base_model = model_name.replace("-high", "").replace("-low", "")
            reasoning_effort = "high" if "high" in model_name else "low" if "low" in model_name else None
            return base_model, reasoning_effort, True
            
        else:
            # 기존 GPT 모델들
            return model_name, None, False

    def predict_batch(self, prompts):
        completions = []
        base_model, reasoning_effort, use_completion_tokens = self._get_model_config(self.model)
        
        # 배치 처리 설정
        if use_completion_tokens:
            batch_completion_kwargs = {"max_completion_tokens": self.max_completion_tokens}
            if reasoning_effort:
                batch_completion_kwargs["reasoning_effort"] = reasoning_effort
        else:
            batch_completion_kwargs = {"max_tokens": self.max_tokens, "temperature": self.temperature}
        
        messages = batch_completion(
            model=base_model,
            messages=prompts,
            **batch_completion_kwargs
        )
        contents = [message['choices'][0]['message']['content'] for message in messages]
        completions.extend(contents)
        return completions


    def predict_single(self, prompts):
        completions = []
        base_model, reasoning_effort, use_completion_tokens = self._get_model_config(self.model)
        
        for prompt in prompts:
            if use_completion_tokens:
                kwargs = {
                    "model": base_model,
                    "messages": prompt,
                    "max_completion_tokens": self.max_completion_tokens
                }
                if reasoning_effort:
                    kwargs["reasoning_effort"] = reasoning_effort
            else:
                kwargs = {
                    "model": base_model,
                    "messages": prompt,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature
                }
            
            message = completion(**kwargs)
            completions.append(message['choices'][0]['message']['content'])
        
        return completions
    
    def predict_batch_openai(self, prompts):
        completions = [None] * len(prompts)
        id_list = [f"request-{i}" for i in range(len(prompts))]
        base_model, reasoning_effort, use_completion_tokens = self._get_model_config(self.model)
        
        # OpenAI 배치 처리 설정
        if use_completion_tokens:
            body_kwargs = {"max_completion_tokens": self.max_completion_tokens}
            if reasoning_effort:
                body_kwargs["reasoning_effort"] = reasoning_effort
        else:
            body_kwargs = {"max_tokens": self.max_tokens, "temperature": self.temperature}

        response_dict = openai_utils.process_batch(
            self.client, 
            base_model, 
            id_list, 
            prompts, 
            job_description=self.job_description, 
            body_kwargs=body_kwargs
        )
        
        for id, response in response_dict.items():
            completions[int(id.split("-")[1])] = response
        
        return completions
    
    def predict(self, dataset, batch_size=1, use_openai_batch=False, completed_batch_id=None):
        if completed_batch_id:
            response_dict = openai_utils.retrieve_batch_responses(self.client, completed_batch_id)
            dataset_dict = {item["id"]: item for item in dataset}
            for id, response in response_dict.items():
                index = int(id.split("-")[1])
                item = dataset_dict[index]
                yield item, response

            return
        
        if use_openai_batch:
            assert batch_size > 1, "Batch size must be greater than 1 for OpenAI batch completion"
            predict_func = self.predict_batch_openai
        else:
            predict_func = self.predict_batch if batch_size > 1 else self.predict_single
        
        # batch iteration
        batch = []
        for i, item in enumerate(dataset):
            prompt = self.build_prompt(item)
            if prompt is None:
                continue
            
            batch.append(prompt)
            if len(batch) == batch_size:
                predictions = predict_func(batch)
                for j, prediction in enumerate(predictions):
                    yield dataset[i - batch_size + j + 1], prediction
                batch = []
                break


        if batch:
            predictions = predict_func(batch)
            for j, prediction in enumerate(predictions):
                yield dataset[i - len(batch) + j + 1], prediction
            batch = []


class RAGRecipePredictor(RecipePredictor):

    def __init__(self, model="gpt-4o-mini", batch_size=1, max_tokens=4096, max_completion_tokens=16384, temperature=0, api_key=None, prompt_filename = "prompts/rag.txt",
                 rag_topk: int = 5, retrieval_split: str = "all"):
        super().__init__(model, batch_size, max_tokens, max_completion_tokens, temperature, api_key, prompt_filename)
        self.job_description = f"RAG material prediction job w/ {model}"
        
        # 데이터셋 이름 변경
        if retrieval_split == "all":
            retrieval_set = load_dataset("iknow-lab/oxidesc-recipe-embeddings", token= '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')
            self.retrieval_set = concatenate_datasets(retrieval_set.values())
        else:
            self.retrieval_set = load_dataset("iknow-lab/oxidesc-recipe-embeddings", split="train", token= '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')
        
        self.rag_topk = rag_topk
        # assert self.rag_topk > 0, "RAG topk must be greater than 0"

        # FAISS 인덱스 처리 개선
        faiss_name = f"faiss_index_{retrieval_split}.faiss"
        try:
            if os.path.exists(faiss_name):
                self.retrieval_set.load_faiss_index("contributions_embedding", faiss_name)
                print(f"Loaded existing FAISS index: {faiss_name}")
            else:
                print("Creating new FAISS index...")
                self.retrieval_set.add_faiss_index("contributions_embedding", "contributions_embedding")
                self.retrieval_set.save_faiss_index("contributions_embedding", faiss_name)
                print(f"Saved FAISS index: {faiss_name}")
        except Exception as e:
            print(f"FAISS index error: {e}")
            print("Creating FAISS index without saving...")
            self.retrieval_set.add_faiss_index("contributions_embedding", "contributions_embedding")

        self.base_references = None

    def search(self, contribution, k=5, return_rows=False):
        query_embedding = np.array(contribution)
        scores, results = self.retrieval_set.get_nearest_examples("contributions_embedding", query_embedding, k)
        if return_rows:
            return results  
        else:
            retrieval_prompts = []
            for i, (contribution, recipe) in enumerate(zip(results["contribution"], results['recipe'])):
                retrieval_prompts.append(f"# Reference {i + 1}:\n{contribution}\n\n{recipe}")
            retrieval_prompts = "\n\n".join(retrieval_prompts)
            return retrieval_prompts

    def build_prompt(self, item):
        contributions, recipe, embeddings = item["contribution"], item["recipe"], item["contributions_embedding"]
        if self.rag_topk > 0:
            retrieval_prompts = self.search(embeddings, k=self.rag_topk)
        else:
            retrieval_prompts = ""

        if self.base_references:
            references = [f"# User Provided Reference {i + 1}:\n{recipe}" for i, recipe in enumerate(self.base_references)]
            retrieval_prompts = "\n\n".join(references) + "\n\n" + retrieval_prompts

        prompt = self.prediction_prompt.format(contributions=contributions, references=retrieval_prompts)
        return [
            {
                "content": prompt,
                "role": "user"
            }
        ]

def main(
        model: str = "gpt-4o-mini",
        prompt_name: str = "prediction",
        batch_size: int = 1,
        use_openai_batch: bool = False,
        use_rag: bool = False,
        top_k: int = 5,
        split: str = "test_high_impact",
):
    ds = load_from_disk("data/omg", split=split)

    model_name = model.split("/", 1)[-1]

    if use_rag:
        if prompt_name == "prediction":
            prompt_name = "rag"
            print("Using RAG prompt instead")
        
        prompt_filename = f"prompts/{prompt_name}.txt"
        predictor = RAGRecipePredictor(model=model, prompt_filename=prompt_filename, rag_topk=top_k)
        output_filename = f"data/{split}/{model_name}/{prompt_name}__k{top_k}.jsonl"

    else:
        prompt_filename = f"prompts/{prompt_name}.txt"
        predictor = RecipePredictor(model=model, prompt_filename=prompt_filename)
        output_filename = f"data/{split}/{model_name}/{prompt_name}.jsonl"


    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    if os.path.exists(output_filename):
        skip = len(list(jsonlines.open(output_filename)))
        ds = ds.select(range(skip, len(ds)))
        print(f"Skipping {skip} items")

    with jsonlines.open(output_filename, "a") as fout:
        for item, prediction in tqdm(predictor.predict(ds, batch_size=batch_size, use_openai_batch=use_openai_batch), total=len(ds)):
            item["prediction"] = prediction
            if 'contributions_embedding' in item:
                del item['contributions_embedding']
            fout.write(item)

    return output_filename

if __name__ == "__main__":
    fire.Fire(main)
