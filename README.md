# Recommendation-system

## Introduction
Developed various algorithms to make recommendations for movies below are the following algorithms.
*  User-Based Collaborative Filtering Algorithms
*  Item-Based Collaborative Filtering Algorithm 
*  Hybrid Algorithm

## Train Data
A set of movie ratings by 200 users (userid: 1-200) on 1000 movies (movieid: 1-1000). The data is stored in a 200 row x 1000 column table. Each row represents one user. Each column represents one movie. A rating is a value in the range of 1 to 5, where 1 is "least favored" and 5 is "most favored"

## Test Data
A pool of movie ratings by 100 users (userid: 201-300,text5.txt). Each user has already rated 5 movies. The format of the data is as follows: the file contains 100 blocks of lines. Each block contains several triples : (U, M, R), which means that user U gives R points to movie M.Note that in the test file, if R=0, then you are expected to predict the best possible rating which user U will give movie M.

## Result
In our Analysis Item-Based Collaborative Filtering Algorithm Performed better than User-Based Collaborative Filtering Algorithms by 10% and Hybrid Alogrith where we used the results User-based and Item-based in ratio of 40% and 60% our results improved by 5% more.
