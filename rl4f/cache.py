from dataclasses import dataclass
from typing import List, Dict
import json
import os

DEFAULT_MAX_TOKENS = 512

@dataclass
class AspectCacheKey:
    scenario: str
    aspect: str

    def __hash__(self):
        return hash((self.scenario, self.aspect))
    
    def __eq__(self, other):
        return hash(self) == hash(other)
    
    
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
    def __init__(self, cache_file='cache.jsonl', save_every=100):
        self.cache_file = cache_file
        self.updates_since_last_saved = 0
        self.save_every = save_every
        self.load_cache()


    def load_cache(self):
        self.cache = {}
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r') as f:
                for line in f.readlines():
                    j = json.loads(line)
                    key = AspectCacheKey.from_json(j)
                    value = AspectCacheValue.from_json(j)
                    self.cache[key] = value

    def save_cache(self):
        with open(self.cache_file, 'w') as f:
            for key, value in self.cache.items():
                j = {
                    **key.to_json(),
                    **value.to_json()
                }
                f.write(json.dumps(j) + '\n')

    def get(self, key: AspectCacheKey):
        return self.cache.get(key, None)
    
    def set(self, key: AspectCacheKey, value: AspectCacheValue):
        self.cache[key] = value
        self.updates_since_last_saved += 1
        if self.updates_since_last_saved >= self.save_every:
            self.save_cache()
            self.updates_since_last_saved = 0