from collections import Counter
import os
import json

PARSED_HISTORY_DIR = "parsed_histories"

# players_to_num_plays = Counter()

# for filename in os.listdir(PARSED_HISTORY_DIR):
#     print filename
#     full_name = PARSED_HISTORY_DIR + "/" + filename
#     with open(full_name) as f:
#         data = json.loads(f.read())
#         for hand in data:
#             for player in hand['players']:
#                 players_to_num_plays[player] += 1

def histogram(num_buckets, bucket_width, c):
    buckets = [0] * num_buckets
    for _, count in c.iteritems():
        if int(count / bucket_width) < num_buckets:
            buckets[int(count / bucket_width)] += 1
    return buckets
