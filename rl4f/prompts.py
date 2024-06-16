generate_aspect_scores_llama = """
You are given a scenario and a schema containing a set of aspects to assess its morality.

Task:

For each aspect, assign an integer score from the set [-4, -3, -2, -1, 0, 1, 2, 3, 4].

Positive Scores [1 to 4]: The aspect was positively evaluated (e.g., 1 = low positive evaluation, 4 = high positive evaluation).
Negative Scores [-4 to -1]: The aspect was negatively evaluated (e.g., -1 = low negative evaluation, -4 = high negative evaluation).
Neutral Score [0]: The scenario is neutral for this aspect.

Provide the scores in strict JSON format with each aspect as the key and the assigned integer score as the value. Ensure the keys correspond to the actual schema aspects provided and do not start with "aspect."
Return only a single JSON object and nothing else.
{
"aspect1": 2,
"aspect2": -3,
...
}

Scenario: {{scenario}}
Aspects: {{aspects}}
"""

generate_morality_score_llama = """
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
  Return only a single JSON object and nothing else.
  
  {
      "morality_score": "... integer score in the range -4 to 4"
  }
"""

generate_morality_score_mixtral = """
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

    Response Format (no explanation required, only the following JSON and nothing else):
    {
        "morality_score": "... integer score in the range -4 to 4"
    }
"""


generate_morality_score_llama_mistral = """
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

            Response Format (no explanation required, only the following JSON and nothing else):
            {
                "morality_score": "... integer score in the range -4 to 4"
            }
"""

get_morality_scores_gpt = """
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
    Return only a single JSON object and nothing else.
    
    {
        "morality_score": "... integer score in the range -4 to 4"
    }
"""