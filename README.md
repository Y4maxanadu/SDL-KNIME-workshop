SDL-KNIME-workshop

for SDL internship and related work with KNIME

RESEARCH BACKGROUND

The purpose of this research is to explore the analysis of geographic information in datasets used for recommendation systems(mainly on Gowalla). Specifically, we aim to investigate how the inclusion of geographic data,  processed and visualized through the KNIME workflow platform, can enhance the interpretability of recommendations and reveal underlying patterns within these platforms beside enhancing the accuracy and effectiveness. The significance of this research lies in its potential to improve the transparency and explainability of recommendation systems, enabling users to better understand the reasoning behind the recommendations and uncover interesting trends and insights.

In our previous research, we had already implemented the recommendation algorithm based on GCN(Graph convolutional neural networks) collaborative filtering and made modified to it in order to preform better on small-scaled datasets. In this Spatial Data Lab internships program, we are introduced with the advanced workflow data analytic platform - KNIME. Prior academic endeavors had already made some practical progress in combining geographical data with recommendation algorithms, however, there is still a need of analyzing the combination of geographical information with advanced recommendation algorithm based on GCN. This research will focus on this topic and utilize practical analytic skills with KNIME platform.

WORKFLOW DEMO

![image](https://github.com/user-attachments/assets/5b127652-d312-432e-aee8-a2b16bcb477b)

Recommending Dataset Gowalla.

Contains users social networks and check-in status geographical data().

Latitude and Longitude of all the check-in point.

Represent the item in interaction graph.

QUICK START WITH KNIME WORKFLOW

1. Download the data processing and visualizing workflow on KNIME Community HUB
KNIME Workflow link: https://hub.knime.com/-/spaces/-/~IhGFsKkfL8H5jkZE/current-state/
2. Import the workflow to your local repository
KNIME Analytics Platform -> File -> Import KNIME Workflow -> Select File
Select file that end with .kmwf
3.  Load data to the workflow
CSV reader node can load the data from csv file, we use to CSV reader node to load user-rating data and a targeted user's social network relationship.
For user-rating data, we initialzed those highly rated checkin place as a pair <user_id , item_id> and there are also the geospatial information about it.(Such as latitude and longitude)
For user social network relationship, we seperate the target user's social relationship into three degree, which represent one-hop neighbor, two-hop neighbor and three-hop neighbor respectively. One-hop neighbor may be the target user's close and direct friend and family members, they have direct relationship and contact in social network. Two-hop neighbor means a person(user on the social network) who is indirectively connected with the target user. This might be a friend of friend and so on. So there goes the three-hop neighbor. We want these three kind of user contributes different weight on effecting the target user recommendation basis.
o1.csv is the data source of User Checkin node(User-rating data)
e2.csv is the data source of User Neighborhood Distance Count node(Target user's social network relationship data)
4. Execute the workflow and present
Right click the Geospatial View Node 'View 0' -> Execute and Open view

![image](https://github.com/user-attachments/assets/2ffc0a80-0938-49ba-ac03-007105f7d4e1)

EXPECTED RESULT

In this part, we will try to demonstrate our recommending works on Geo-data-based dataset by KNIME and try to implement the visualization of the recommending. By utilized the techniques in deep learning and GIS and also the assistance of the KNIME workflow platform, we are expected to accomplish the following work.

(1) Recommendation based on user preference and geographical social relationships.

(2) Visualization of the recommending result by KNIME platform.

(3) Interpretable recommending based on geographical information.

(4) Comparison between the traditional recommendation method and the novel algorithm in specific datasets(Yelp and Gowalla). Achieving better recommending accuracy and ranking scores.

