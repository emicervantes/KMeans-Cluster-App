import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Part 1: Turn the above into a Streamlit app and make the following changes
# Give the app a title related to K-Means
# Include a slider that lets the user choose the number of iterations

st.title("K-Means Cluster App")
iterations = st.slider("Choose the number of iterations: ",1,100)

# we use make_blobs from sklearn to generate random data.
# We specify explicit starting points.
# We specify that the algorithm should only run the procedure one time; that is from the max_iter = 1 step.
# We also plot in black the 5 points used to assign the clusters. 
X, _ = make_blobs(n_samples=1000, centers=5, n_features=2, random_state = 1)
df = pd.DataFrame(X, columns = list("ab"))
starting_points = np.array([[0,0],[-2,0],[-4,0],[0,2],[0,4]])
kmeans = KMeans(n_clusters = 5, max_iter=iterations, init=starting_points, n_init = 1)
kmeans.fit(X)
df["c"] = kmeans.predict(X)
chart1 = alt.Chart(df).mark_circle().encode(
    x = "a",
    y = "b",
    color = "c:N"
)

df_centers = pd.DataFrame(kmeans.cluster_centers_, columns = list("ab"))

chart_centers = alt.Chart(df_centers).mark_point().encode(
    x = "a",
    y = "b",
    color = alt.value("black"),
    shape = alt.value("diamond"),
)




chart1 + chart_centers