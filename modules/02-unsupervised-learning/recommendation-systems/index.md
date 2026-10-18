# Recommendation Systems

A recommendation system answers one question: **what should we show this person next?**

Amazon suggests products. Netflix suggests shows. Spotify suggests songs. Behind all of them are a few ideas that build on each other. These three notebooks go through them in order.

## The three parts

1. **[Market Basket Analysis](../market-basket-analysis/notes.ipynb)**
   Look at what people buy together. Rules like "people who buy jeans also buy belts". Same advice for everybody. Tools: support, confidence, lift, Apriori.

2. **[Collaborative Filtering](../recommender-systems/notes.ipynb)**
   Now make it personal. Use the user–item rating table to find similar users or similar items. Also covers cold start and content-based recommendations.

3. **[Matrix Factorization](../matrix-factorization/notes.ipynb)**
   Learn a short hidden profile for every user and every item, and recommend by matching them. This is the idea behind the Netflix Prize. It also connects K-Means, GMM and NMF into one picture.

## How to read them

Go in order. Each part fixes a weakness of the one before it: rules are not personal, and similarity methods break when the rating table is mostly empty. Matrix factorization is the answer to that.
