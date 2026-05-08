Mark Evers  
5/7/2026  
CSCI 4930 - Machine Learning  

# Homework 5  

### Task 4
> After analyzing the plot, the weak learners become much less accurate after the first round.  This is expected behavior as the model focuses on the predictions that it got wrong by weighting those samples higher as time goes on.
> The ensemble model starts off at the same accuracy as the first weak learner, and then declines and plateaus as the weak learners become less accurate due to overfitting.  The outliers in the training data are likely causing the weak learners to become less accurate as they focus more and more on those outliers, which may not be representative of the overall data distribution.  The baseline model performs better than the ensemble model after a certain number of rounds, which suggests that the ensemble is overfitting to the training data and not generalizing well to the test data.
> I would like to see what happens if I were to use decision trees as the weak learners instead of logistic regression, as decision trees are more commonly used as weak learners in AdaBoost and may be less prone to overfitting in this case.
