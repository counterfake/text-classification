from counterfake_api_lib.authentication import AuthenticatedSession
import pandas as pd
s = AuthenticatedSession()

results = pd.read_csv("results3.csv")
for d in tqdm(results):
    print(d)
    break
    r = s.request("PATCH", "https://api.counterfake.ai/products/{}".format(d['id']), json = {'category' : d['category']})
