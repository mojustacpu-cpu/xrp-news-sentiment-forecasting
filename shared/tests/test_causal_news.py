from pathlib import Path
import sys,unittest
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'shared/feature_engineering'))
from causal_news import aggregate
def article(times,sentiments=None,relevance=None):
 n=len(times);return pd.DataFrame({'published_at_utc':times,'sentiment':sentiments if sentiments is not None else [1.]*n,'xrp_specificity':relevance if relevance is not None else [1.]*n,'stance_xrp':[1.]*n,'uncertainty':[.2]*n})
class CausalTests(unittest.TestCase):
 def test_future_is_excluded(self):
  t=pd.Timestamp('2025-01-01T12:00Z');base=article([t]);extended=article([t,t+pd.Timedelta(seconds=1)],[1.,-1.]);pd.testing.assert_frame_equal(aggregate(base,[t]),aggregate(extended,[t]))
 def test_completed_hour_boundary(self):
  d=article(['2025-01-01T12:01Z']);out=aggregate(d,pd.to_datetime(['2025-01-01T12:00Z','2025-01-01T13:00Z']))
  self.assertEqual(out.eligible_articles.tolist(),[0,1])
 def test_half_life_and_normalized_average(self):
  t=pd.Timestamp('2025-01-01T00:00Z');out=aggregate(article([t]),[t,t+pd.Timedelta(hours=12)])
  self.assertAlmostEqual(out.relevance_weight_sum.iloc[1],.5*out.relevance_weight_sum.iloc[0]);self.assertAlmostEqual(out.equation_sentiment.iloc[1],1.,places=8)
 def test_empty_and_irrelevant(self):
  t=pd.Timestamp('2025-01-01T00:00Z');out=aggregate(article([t],relevance=[0.]),[t]);self.assertEqual(out.no_relevant_news.iloc[0],1);self.assertEqual(out.eligible_news_present.iloc[0],1);self.assertEqual(out.equation_sentiment.iloc[0],0)
  empty=aggregate(article([]),[t]);self.assertEqual(empty.eligible_articles.iloc[0],0);self.assertEqual(empty.no_relevant_news.iloc[0],1)
 def test_window_boundary(self):
  t=pd.Timestamp('2025-01-01T00:00Z');out=aggregate(article([t]),[t+pd.Timedelta(hours=48),t+pd.Timedelta(hours=49)]);self.assertEqual(out.eligible_articles.tolist(),[1,0])
 def test_delay_and_nonnegative_age(self):
  t=pd.Timestamp('2025-01-01T00:00Z');out=aggregate(article([t]),[t,t+pd.Timedelta(hours=6)],delay=6);self.assertEqual(out.eligible_articles.tolist(),[0,1]);self.assertEqual(out.relevance_weight_sum.iloc[1],1.)
 def test_relevance_weighting(self):
  t=pd.Timestamp('2025-01-01T00:00Z');out=aggregate(article([t,t],[1.,-1.],[1.,.5]),[t]);self.assertAlmostEqual(out.equation_sentiment.iloc[0],1/3,places=8)
 def test_deterministic_order(self):
  t=pd.Timestamp('2025-01-01T00:00Z');d=article([t,t+pd.Timedelta(hours=1)],[1.,-.5]);pd.testing.assert_frame_equal(aggregate(d,[t]),aggregate(d,[t]))
 def test_reject_negative_delay(self):
  with self.assertRaises(ValueError):aggregate(article([]),[],delay=-1)
if __name__=='__main__':unittest.main()
