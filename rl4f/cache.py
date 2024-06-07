from dataclasses import dataclass
from typing import List, Dict
import json
import os

DEFAULT_MAX_TOKENS = 512

@dataclass
class AspectCacheKey:
    scenario: str
    aspect: str

    def __hash__(self) -> int:
        return hash((self.scenario, self.aspect))
    
    def to_json(self):
        return {
            "scenario": self.scenario,
            "aspect": self.aspect
        }
    
    @staticmethod
    def from_json(json):
        return AspectCacheKey(
            scenario=json["scenario"],
            aspect=json["aspect"]
        )

@dataclass
class AspectCacheValue:
    response: str
    
    def __hash__(self):
        return hash(self.response)

    def to_json(self):
        return {
            "response": self.response
        }
    
    @staticmethod
    def from_json(json):
        return AspectCacheValue(
            response=json["response"]
        )


class AspectCache:
    def __init__(self, cache_file='cache.jsonl', save_every=100, log_hit_rate_every=1000):
        self.cache_file = cache_file
        self.updates_since_last_saved = 0
        self.save_every = save_every
        self.log_hit_rate_every = log_hit_rate_every
        self.hits = 0
        self.calls = 0
        self.load_cache()


    def load_cache(self):
        self.cache = {}
        # if os.path.exists(self.cache_file):
        #     with open(self.cache_file, 'r') as f:
        #         for line in f.readlines():
        #             j = json.loads(line)
        #             key = AspectCacheKey.from_json(j)
        #             value = AspectCacheValue.from_json(j)
        #             self.cache[key] = value

    def save_cache(self):
        with open(self.cache_file, 'w') as f:
            for key, value in self.cache.items():
                j = {
                    **key.to_json(),
                    **value.to_json()
                }
                f.write(json.dumps(j) + '\n')

    def get(self, key: AspectCacheKey):
        self.calls += 1 

        if self.calls % self.log_hit_rate_every == 0:
            hit_rate = self.hits / self.calls * 100
            print(f'\n\nCache Hit Rate: {hit_rate:.2f} %')

        if key in self.cache:
            self.hits += 1
            return self.cache.get(key)
        return None
    
    def set(self, key: AspectCacheKey, value: AspectCacheValue):
        # self.cache[key] = value
        self.updates_since_last_saved += 1
        if self.updates_since_last_saved >= self.save_every:
            # self.save_cache()
            self.updates_since_last_saved = 0