# LT2222 V21 Assignment 2

Assignment 2 from the course LT2222 in the University of Gothenburg's winter 2021 semester.

Your name: Anna Page

## Part 5 - Evaluation

- Comparing the performance on the training data against the testing data, the model is clearly overfitting. The overall performance on the test data is significantly worse.
- There is also significant class imbalance present in the data and the model behaves accordingly. The model performs better on the larger classes and overlooks the smaller classes. For instance, 'art', 'eve', and 'nat' are represented by 53, 45, and 20 data points, respectively. By contrast, 'geo', the largest class, contains 2070 data points. The differences in test performance given this disparity are not surprised.

*Answer all questions in the notebook here.  You should also write whatever high-level documentation you feel you need to here.*

## Bonus Part A - Error analysis

The lowest performing class on the test set is 'eve'. This class has 13 instances in the test set, of which none were correctly classified. The extremely small size of this class relative to the more populous classes is a likely cause of this poor performance. Looking at the most commonly occuring words within these 13 data points, the vast majority are grammatical ('the', 'in', 'and') and very few are lexical. While grammatical words are likely to have similar distributions across all classes, the distribution of lexical words would be more informative. The small sample size and absence of repeated lexical words in this class isn't sufficient for patterns in the data to become pronounced.

## Bonus Part B - Expanding the feature space

To add part of speech features to the model, I altered the Instance class to have another 'pos' attribute which contains the part of speech for each feature in the 'features' attribute. In the case of no part of speech (e.g. a word or a start/end tag), a placeholder ('<NT>') was used. From this part of speech list, an additional four features were calculated: the number of nouns, verbs, adjectives, and adverbs present in that sentence. This was used to create a second 'pos_df' which was concatenated to the original 'df'. The remainder of the process was the same: the data was split with the same random seed, a model was trained, and two confused matrices were calculated. Inspecting the matrices, the differences are minor, but it would appear that the additional features have improved model performance.