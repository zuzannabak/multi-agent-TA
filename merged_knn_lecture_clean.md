# Part I - September 3, 2025: supervised learning and the intuition behind KNN

## 1. The supervised-learning dataset: examples, features, and labels

### Written and visual content from the notes (PDF page 1)

The notes define a labeled dataset as

\[
D=\{(x_i,y_i),\ i=1,\ldots,N\}.
\]

Each input is a feature vector

\[
x_i=(x_{i1},x_{i2},\ldots,x_{iM}).
\]

The annotations identify:

- \(N\): number of examples;
- \(M\): number of features;
- \(x_i\): the complete feature vector for example \(i\);
- \(x_{ij}\): the value of feature \(j\) for example \(i\);
- \(x_j\): feature \(j\), when the particular example does not need to be specified;
- \(y_i\): the label of example \(i\).

The page also uses **features** and **attributes** as equivalent terms.

### Current-year spoken explanation

The instructor said that any machine-learning discussion must begin with the dataset. He chose \(D\) simply because it stands for “dataset” and described it as a set of pairs \((x_i,y_i)\). There are \(N\) examples, and each \(x_i\) contains \(M\) feature values. The associated \(y_i\) is the label for that example.

He connected the notation to a table: the examples can be understood as rows and the features as columns. The full vector \(x_i\) is one row's input information, while a quantity such as \(x_{74}\) means the value of the fourth feature for the seventh example.

### Student questions and notation clarifications

Students asked whether \(x_1\) and \(x_2\) in a formula were examples or variables. The instructor clarified that they are two different **features**. He established the course convention that \(x\) denotes inputs or features and \(y\) denotes the label.

He further distinguished the common notations:

- \(x_i\): example \(i\), represented by its whole feature vector;
- \(x_j\): feature \(j\), when the row is not being specified;
- \(x_{ij}\): feature \(j\) of example \(i\).

A student connected \(x_{74}\) to row 7, column 4 of the earlier automobile table, and the instructor confirmed that interpretation.

---

## 2. Where the dataset comes from: sampling the world

### Written and visual content from the notes (PDF page 1)

A central diagram shows:

1. a cloud labeled **World**;
2. an arrow labeled **sampling** from the world to dataset \(D\);
3. an arrow labeled **learning** from \(D\) to a model \(f\);
4. a new input \(x\) entering \(f\), which produces \(y=f(x)\);
5. an arrow labeled **apply** returning the learned model to new cases in the world.

The diagram emphasizes that the dataset is a finite sample from a broader domain and that the learned model is ultimately intended for use beyond the stored observations.

### Current-year spoken explanation

The instructor said that a dataset never appears from nowhere: it comes from a larger “world.” For the automobile data used earlier in class, the relevant world was the population of cars, and the dataset consisted of sampled car models sold in the United States between roughly 1970 and 1982.

He stressed that sampling is a major topic that could occupy an entire course. Whenever someone receives a dataset, they should ask where it came from and how it was collected. The lecture would usually assume that the data are a random or reasonably representative sample from an underlying distribution, but that assumption should not be forgotten.

### Survey example and sampling-bias aside

The instructor used public-opinion surveys to show why representative sampling is difficult. A researcher cannot ask every person in the United States, so perhaps only a thousand people are contacted. If that group is genuinely representative, its average or majority opinion may tell us something about the larger population.

However, even apparently simple strategies introduce bias. Randomly calling phone numbers excludes people without phones. Many people will hang up or refuse to participate. The remaining respondents may differ systematically from those who refuse. For example, busy parents may be less likely to complete a survey. The final sample can therefore represent “people willing and able to answer this call” rather than the whole population.

The instructor's practical warning was that analysis and modeling cannot repair every flaw in the original sampling process. Understanding data provenance is essential to understanding what a model's reported performance means.

---

## 3. The supervised-learning loop and the prediction model \(f\)

### Written and visual content from the notes (PDF page 1)

The model is labeled

\[
f:\text{ prediction model}.
\]

Its input-output relationship is written as

\[
y=f(x).
\]

A boxed statement summarizes the task:

> Given \(D\), learn \(f\) that accurately predicts \(y\) from \(x\).

### Current-year spoken explanation

Once the dataset has been sampled, supervised learning attempts to learn a function mapping the feature vector to the label. The instructor described this as finding the relationship between the \(M\) features and \(y\), with the requirement that the resulting mapping be accurate.

After learning, the function is returned to the world and used on new cases. The instructor said that he would usually call \(f\) a **prediction model**, or simply a **model**. The full process is therefore:

1. sample a dataset from the world;
2. learn a model from that dataset;
3. apply the model to new examples in the world.

### Spoken examples and classroom questions

The main example was a spam filter. An incoming email must first be represented as a feature vector \(x\). The function then predicts whether it is spam. A useful filter must avoid both letting too much spam into the inbox and incorrectly sending legitimate messages to the spam folder.

A student asked whether the label in the car dataset could be the name of the car model. The instructor said it technically could be, because a supervised-learning target is defined by the task, but that the dataset was more plausibly intended to predict miles per gallon. He added that a modern system such as ChatGPT could itself be viewed as a very complicated function \(f\): it takes an input and returns an output, although its output may be a sentence rather than one scalar label.

The instructor explained that, in course exercises, the target \(y\) would normally be specified. In a real project, the practitioner would need to talk with a customer or domain expert to determine what the actual prediction task should be.

---

## 4. The three organizing questions of supervised learning

### Written and visual content from the notes (PDF page 1)

The page organizes supervised learning around three questions:

1. **What is \(f\)?**
2. **How good is \(f\)?**
3. **How do we learn \(f\) from \(D\)?**

These questions connect the choice of model, the definition of success, and the learning procedure.

### 4.1 What is \(f\)?

#### Written content

The notes first show a mathematical model:

\[
f(x)=3+2x_1-7x_2,
\]

identified as a linear-regression-type function. They note that \(f\) can also be a nonlinear function of the features, leading toward neural networks and deep learning.

The notes then broaden the model concept: \(f\) may be a **computer algorithm**, including

- a **search-based** algorithm such as k-nearest neighbors;
- a **rule-based** algorithm such as a decision tree.

Green annotations indicate the approximate course sequence: KNN first, then linear regression, later neural networks/deep learning, and decision trees near the end.

#### Current-year spoken explanation

The instructor said that a prediction model can be a literal mathematical formula. The example \(3+2x_1-7x_2\) is linear in the two features and belongs to linear regression. A nonlinear function leads naturally toward neural networks and what is now commonly called deep learning.

He then emphasized that \(f\) can be more general than a familiar algebraic expression: it can be a computer program. A program accepts \(x\), performs a procedure, and returns \(y\). This permits inputs that may include numerical, categorical, or string-like information after appropriate representation.

He divided algorithmic models into two intuitive families:

- **search-based:** KNN searches the stored examples for close neighbors;
- **rule-based:** a decision tree applies if-then or if-else decisions, such as checking whether particular features satisfy thresholds before predicting spam.

The choice of what \(f\) should be determines which machine-learning family is being used. An experienced practitioner looks at the task and data and decides whether a linear model, deep-learning model, tree, or another model family is a plausible fit.

### 4.2 How good is \(f\)?

#### Written content

The notes ask whether \(f_1\) is better than \(f_2\). They state that this requires an **accuracy measure** or **loss function**.

#### Current-year spoken explanation

The instructor imagined a company offering a million-dollar prize for the best spam filter. Multiple machine-learning experts could submit competing models, but the company would need a precise rule for deciding which model is best.

An accuracy measure rewards correct predictions. A loss function measures or penalizes mistakes. The desired direction differs - high accuracy and low loss - but both provide a quantitative way to compare models such as \(f_1\) and \(f_2\).

### 4.3 How do we learn \(f\) from \(D\)?

#### Written content

The third question connects the dataset to the creation of the prediction model. The notes use both “learn” and “create.”

#### Current-year spoken explanation

The instructor called this the core machine-learning question, but said it cannot be answered sensibly until the first two are specified. Before designing a learning procedure, one needs to know what kind of object \(f\) is and how its success will be evaluated. Only then can an algorithm search for or construct a good model from \(D\).

---

## 5. Classification versus regression: the visual mental picture

### Written and visual content from the notes (PDF pages 1-2)

The notes identify two major types of supervised learning:

- **classification:** \(y\) is categorical;
- **regression:** \(y\) is numerical, with \(y\in\mathbb{R}\).

Page 2 is headed **Mental Picture of S.L.** and contains two sketches.

#### Binary-classification sketch

A two-dimensional plot uses \(x_1\) and \(x_2\) as features. Red minus signs and green plus signs represent two classes, with a notation such as \(y\in\{+1,-1\}\). Two different candidate boundaries are drawn:

- an irregular orange boundary \(f_1(x)\);
- a curved magenta boundary \(f_2(x)\).

The models divide the same feature space differently and can therefore make different predictions.

#### Regression sketch

A second graph places \(x_1\) on the horizontal axis and \(y\) on the vertical axis. Black observations form a numerical pattern. Two candidate models are shown:

- a magenta straight line \(f_1(x)\);
- a green nonlinear curve \(f_2(x)\).

The picture raises the question of which model follows the data more accurately.

### Current-year spoken explanation

The only essential difference introduced here was the type of label:

- classification predicts a category, such as spam/non-spam or a car color;
- regression predicts a real number.

For binary classification, the instructor imagined a scatterplot with two features. Instead of ordinary dots, each example is drawn as a plus or minus. With more classes, additional shapes such as triangles, rectangles, or stars could be used. He acknowledged that the human visual system cannot directly picture hundreds of features, but said that the two-dimensional representation is still useful for reasoning about algorithms.

For regression, using a separate symbol for every possible real-valued label would be impossible. The instructor therefore simplified to one feature on the horizontal axis and the numerical label \(y\) on the vertical axis. The data remain a scatterplot, but now a candidate function can be drawn through the points.

He compared a relatively poor function with another that passed more centrally through the observations. This made the three earlier questions concrete: why is one function better, what formula measures the difference, and how can the better function be learned rather than merely drawn by eye?

He then returned to classification and drew competing boundaries. One assigned everything above a curve to one class and everything below it to the other; another separated the regions differently and appeared to make more correct classifications.

### Student question about mathematical assumptions

A student asked whether mathematical models are assumed to be continuous, differentiable, or smooth. The instructor answered that such assumptions are often useful and often made, but not universally. Machine learning always involves assumptions about the data and the function class, and those assumptions depend on the chosen method.

---

## 6. Discovering nearest-neighbor reasoning before naming the algorithm

### Written and visual content from the notes (PDF page 2)

Under **1st S.L. Algorithm**, the notes show red minus examples concentrated in one region and green plus examples in another. A new unlabeled point is placed near several negatives. An annotation says that it should probably be negative because its nearby neighbors are negative.

The page labels the underlying principle:

- **guilt by association**;
- nearby or similar examples are expected to have similar labels.

This intuition is connected to **k-nearest neighbors (KNN)**.

### Current-year spoken explanation

The instructor deliberately asked the class to solve a new classification case before formally presenting an algorithm. He placed a new example in the existing plus/minus plot and asked whether it should be positive or negative.

Most students chose negative. One student argued for positive because an isolated negative already existed on the other side of the plot while no corresponding positive outlier existed. The instructor said that even this dissenting reasoning was relevant: the class was trying to infer the unknown label from the arrangement of nearby observed labels.

When asked to defend the negative prediction, a student said that the nearby examples were negative. The instructor reformulated the reasoning: examine the neighborhood, see which labels live there, and predict accordingly. The prediction is not known with certainty, but it is a plausible guess.

He named the algorithm **k-nearest neighbors**, abbreviated **KNN**, and then asked a deeper question: why should having negative neighbors make the new example negative? The answer was an assumed principle of **guilt by association**. The model treats association with one class as evidence for belonging to that class.

### Analogy and broader claim

The instructor compared this machine-learning assumption with everyday bias: as a first approximation, someone may be judged by the neighborhood or group with which they are associated, even though closer inspection could reveal that the individual is different. In KNN, this is not a moral judgment but a modeling assumption that local similarity carries information.

He said that essentially all machine-learning algorithms rely on some related form of association or similarity. Without a principle connecting new examples to previously observed cases, generalization would be impossible.

### End of the first current-year class

The instructor stopped at the intuition rather than deriving the full KNN procedure. He announced that the following week would develop KNN pseudocode, analyze computational cost, and use the algorithm to introduce additional machine-learning terminology and practical decisions.

---

# Part II - September 10, 2025: the KNN algorithm, evaluation, and dimensionality

## 7. Recording gap: formal KNN setup and prediction rule

### Written and visual content from the notes (PDF page 2)

The setup is:

- given a labeled dataset \(D=\{(x_i,y_i)\}\);
- given a new unlabeled example \(x\);
- objective: predict the label \(y\) of \(x\).

The KNN procedure is written in three steps:

1. calculate the distances between \(x\) and all examples in \(D\);
2. identify the \(k\) examples with the smallest distances;
3. use the labels of those \(k\) neighbors to predict the label of \(x\).

For binary classification, the page specifies **majority voting**.

### Reconstructed spoken explanation from the prior-year lecture

KNN keeps the labeled training examples and uses them directly when a new query arrives. It does not first compress the data into a short fitted formula.

- With \(k=1\), the query receives the label of its single closest training example.
- With \(k=3\), the three closest labels vote. If two are positive and one is negative, the predicted class is positive.

The instructor used the ambiguous point from the neighborhood sketch to motivate why more than one neighbor can be useful: a single nearest example may be noisy or atypical, while a small local group may better reflect the surrounding region.

---

## 8. Euclidean distance, pseudocode, and computational cost

### Written and visual content from the notes (PDF page 2)

For numerical features, the page gives Euclidean distance:

\[
\operatorname{dist}(x,x_i)
=
\sqrt{(x_{i1}-x_1)^2+(x_{i2}-x_2)^2+\cdots+(x_{iM}-x_M)^2}.
\]

The pseudocode is organized into three stages:

```text
# Compute all distances from x
for i = 1, ..., N:
    d(i) = dist(x, x_i)

# Identify the k nearest neighbors
index = sort(d)

# Predict using the neighbor labels
ypred = majority(y(index(1:k)))
```

Sorting must preserve the connection between each distance and the training example from which it came.

The notes annotate the computational cost:

- one distance across \(M\) features: \(O(M)\);
- all \(N\) distances: \(O(NM)\);
- sorting the distances: \(O(N\log N)\);
- voting over \(k\) labels: \(O(k)\).

Therefore,

\[
O(NM)+O(N\log N)+O(k),
\]

which is treated approximately as

\[
O(NM)
\]

when the full distance scan dominates.

### Reconstructed spoken explanation

A single Euclidean distance compares all \(M\) feature coordinates, so its cost grows linearly with \(M\). KNN repeats that work for all \(N\) stored training examples. Afterward, it sorts the distances and takes a vote among the first \(k\) labels.

The central computational fact is not the exact constant in front of the formula. It is that the basic algorithm scans almost the entire dataset for every prediction. As the number of examples or features grows, query-time prediction becomes costly.

The pseudocode was presented not only as an implementation guide but also as a way to expose the bottleneck. KNN is simple to “train” because the main action is storing the data, but that simplicity is paid for when a new prediction requires a large search.

---

## 9. Hyperparameters and the problem of choosing \(k\)

### Written and visual content from the notes (PDF page 2)

A box labeled **Practical Questions** asks:

- What is \(k\)?
- What is \(\operatorname{dist}(\cdot)\)?

The page calls such prior decisions **hyperparameters**. A note beneath \(k\) says that it should be neither too small nor too large and that selecting it requires trial and error.

### Reconstructed spoken explanation

A very small value, especially \(k=1\), makes a prediction depend completely on one training example. This can create unstable predictions around outliers or mislabeled observations.

A very large value includes examples that are too far away to be meaningful neighbors. As \(k\) approaches the dataset size, local structure disappears and the global majority class dominates.

The practical intuition is therefore:

- do not choose \(k\) so small that one noisy point controls the prediction;
- do not choose \(k\) so large that distant observations overwhelm the local neighborhood;
- compare several candidate values empirically.

A radius-based alternative is possible: include every example within a chosen distance threshold. However, this replaces one hyperparameter with another. A radius that is too small may return no neighbors; a radius that is too large includes irrelevant examples.

The prior-year discussion also stressed that dataset size alone does not guarantee useful neighbors. What matters is whether examples are actually close in the feature space - an issue that later becomes central in the curse-of-dimensionality discussion.

---

## 10. Distance measures and the point where the 2025 recording resumes

### Written and visual content from the notes (PDF page 3, upper section)

The notes list several distance choices:

- Euclidean distance;
- the general \(L_p\)-norm or Minkowski family;
- Manhattan distance when \(p=1\);
- cosine distance;
- Hamming distance for binary features.

The general formula is

\[
\operatorname{dist}(x,x_i)
=
\left(
|x_{i1}-x_1|^p+
|x_{i2}-x_2|^p+
\cdots+
|x_{iM}-x_M|^p
\right)^{1/p}.
\]

Special cases include:

- \(p=2\): Euclidean or \(L_2\) distance;
- \(p=1\): Manhattan or \(L_1\) distance;
- \(p\to\infty\): the largest coordinate-wise difference determines the distance.

A small grid sketch illustrates Manhattan movement along coordinate directions. For binary vectors, the page marks positions where two bit strings disagree and counts the mismatches.

### Reconstructed portion before the recording resumes

The prior-year lecture introduced Euclidean distance as the usual default for numerical features and placed it inside the broader \(L_p\) family. Manhattan distance changes the geometry by adding absolute coordinate differences. Cosine distance compares vector direction or angle and can be useful when direction matters more than raw magnitude.

The instructor's main point was not that students should always prefer an exotic metric, but that “nearest” has no meaning until distance has been defined. The metric must fit the representation and the application.

### Current-year recording resumes: Hamming distance

The 2025 recording begins in the middle of a binary-vector comparison. The instructor counted positions where the vectors disagreed and initially announced a distance of 3. A student noticed another mismatch. The instructor corrected the Hamming distance to **4** and confirmed that Hamming distance is simply the number of disagreements between binary features.

He then summarized the practical difficulty: even in an apparently simple algorithm, the practitioner must decide what \(k\) means, which metric to use, and how to handle mixtures of numerical, binary, and categorical features.

### Current-year classroom questions about distance choice

A student asked why one would use a different \(L_p\) distance if Euclidean and the alternatives all have the same asymptotic cost for one pair of examples. The instructor corrected the relevant complexity to \(O(M)\) per distance and said that changing the norm generally does not reduce that computational order.

His practical answer was that Euclidean distance is the usual default unless the data or domain provides a reason to choose something else. He could not recall often using Manhattan distance in his own work, though he emphasized that it remains a legitimate option.

Another student asked about \(L_\infty\). The instructor explained that it selects the largest individual coordinate difference: among the distances along all dimensions, the maximum one becomes the overall distance. He could not immediately offer a routine KNN use case, but said that norms would reappear later in the course in other important contexts.

---

## 11. Feature scaling and normalization

### Written and visual content from the notes (PDF page 3)

The page asks what happens when features have very different scales. Its example uses:

- \(x_1\): CPU size, on a scale around \(10^{-3}\);
- \(x_2\): CPU speed or clock frequency, on a scale around \(10^9\).

A sketch shows points compressed near one axis because the larger-scale feature dominates the geometry. The note says that distance becomes dominated by \(x_2\).

Two transformations are written.

#### Min-max scaling

\[
x'_j=
\frac{x_j-\min(x_j)}
{\max(x_j)-\min(x_j)},
\qquad x'_j\in[0,1].
\]

#### Standardization

\[
x'_j=
\frac{x_j-\operatorname{mean}(x_j)}
{\operatorname{std}(x_j)}.
\]

After standardization, the transformed feature has mean 0 and standard deviation 1.

The page labels scaling as **data preprocessing**, meaning a change to \(D\) before the learning algorithm is applied.

### Current-year spoken explanation

The instructor imagined a computer-related prediction task. CPU physical size might be recorded in meters or millimeters, giving values near \(10^{-3}\), while clock speed measured in hertz may be near \(10^9\). In raw coordinates, the second feature overwhelms the first.

If the data are plotted using a scale large enough for clock frequency, all variation in CPU size appears almost at zero. In a Euclidean-distance calculation, \(x_1\) contributes essentially nothing. The algorithm behaves as though the first feature had been discarded, even if it is the more important predictor.

The solution is to rescale **every** feature, not only the small one. Min-max scaling maps each feature to the interval from 0 to 1. Standardization subtracts the feature mean and divides by its standard deviation, producing mean 0 and standard deviation 1.

The instructor described the goal as giving every feature an equal chance - or equal initial importance - in the distance calculation. Scaling does not prove that all features are equally predictive. It prevents arbitrary measurement units from deciding the issue in advance.

### Data-preprocessing terminology

The instructor defined data preprocessing as any change or modification to the dataset before the prediction model is learned. Scaling is one example. He presented “hyperparameter” and “data preprocessing” as terms that sound intimidating at first but refer to concrete modeling decisions students were already making.

---

## 12. Evaluating KNN: split the data first

### Written and visual content from the notes (PDF page 3)

The page shows a labeled dataset \(D\) randomly split into:

- \(D_{\text{train}}\), used exclusively to create \(f\);
- \(D_{\text{test}}\), used exclusively to estimate the accuracy of \(f\).

The illustration uses a roughly two-thirds/one-third split. A second diagram shows \(D_{\text{train}}\) feeding the creation of \(f\), while \(D_{\text{test}}\) is held out and used only for evaluation.

### Current-year spoken explanation

The instructor called data splitting the first essential step in evaluating not only KNN but any machine-learning model. He imagined a customer providing only 50 labeled examples and asking for a predictor. If all 50 are used to construct the model, no independent observations remain to estimate how well it will work on unseen cases.

One theoretical option would be to return to the world and collect more labeled examples, but that may be costly, slow, or impossible. The practical solution is to reserve part of the available dataset before training.

For illustration, the instructor used:

- 35 examples for training;
- 15 examples for testing.

The split should be random. The training subset is used exclusively to build the predictor. The test subset is used exclusively to estimate its accuracy.

For KNN, the 35 training observations are stored. Each of the 15 test observations is then treated as a new query, its neighbors are found among the 35 training examples, and its predicted label is compared with its known true label.

### “Machine-learning jail” and exam analogy

The instructor warned that evaluating a model on the same data used to construct it is a serious error and joked that it is a reason to send a practitioner to “machine-learning jail.” A model may memorize the training examples rather than demonstrate that it can generalize.

He compared this with an exam. Students are not usually given exactly the same questions they completed for homework, because memorizing those answers would not establish broader understanding. Likewise, a model must be tested on separate examples.

### Student question about diversity and sampling

A student asked how a random split guarantees that both subsets contain adequate diversity. The instructor distinguished this from the earlier sampling problem. If the original dataset is biased or unrepresentative, a train-test split cannot fully repair it. Once the dataset is given, the practitioner can mainly promise performance on data similar to that dataset, not on an entirely different population.

---

## 13. Ground truth, predictions, and the confusion matrix

### Written and visual content from the notes (PDF page 3)

The notes show a test table with the true label \(y\) and the prediction \(f(x)\) for each test example. They then aggregate the outcomes into a \(2\times2\) confusion matrix, with true labels across one dimension and predicted labels across the other.

The four counts are:

- **TP - true positive:** true positive, predicted positive;
- **TN - true negative:** true negative, predicted negative;
- **FP - false positive:** true negative, predicted positive;
- **FN - false negative:** true positive, predicted negative.

### Current-year spoken explanation

For each held-out example, there are two values: the known true label and the model's prediction. Looking at a long list of pairs is inconvenient, so the outcomes are summarized in a confusion table, also called a confusion matrix or sometimes a contingency table.

For binary classification, the matrix is \(2\times2\). Each cell is a count, not a percentage. The instructor wrote the true labels along one dimension and predicted labels along the other, then named the four possible outcomes.

During the live explanation, he briefly mixed up the false-positive and false-negative locations while drawing the table, then corrected the intended meanings. The definitions above follow the written notes and the standard interpretation used in the subsequent formulas.

---

## 14. Accuracy, error rate, precision, recall, and F1

### Written and visual content from the notes (PDF page 3)

#### Percent accuracy

\[
\text{Percent accuracy}
=
\frac{TP+TN}{N_{\text{test}}}\times100\%.
\]

#### Error rate

\[
\text{Error rate}
=
1-\frac{TP+TN}{N_{\text{test}}}
=
\frac{FP+FN}{N_{\text{test}}}.
\]

#### Precision

\[
P=\frac{TP}{TP+FP}.
\]

#### Recall

\[
R=\frac{TP}{TP+FN}.
\]

#### F1 score

\[
F1=\frac{2PR}{P+R},
\qquad F1\in[0,1].
\]

A handwritten annotation states that a large F1 score is good.

### Current-year spoken explanation

Accuracy is the proportion of test predictions that are correct. The instructor first derived the fraction \((TP+TN)/N_{\text{test}}\) and then multiplied by 100 when expressing it as a percentage. Error rate is its complement: the fraction of predictions that are wrong.

Precision and recall focus specifically on the positive class:

- **Precision:** among everything predicted positive, what fraction is truly positive?
- **Recall:** among everything that is truly positive, what fraction did the model successfully identify?

The instructor described recall as how many actual positives the model “captured” or recalled, while precision describes how reliable or precise the positive predictions are.

F1 combines precision and recall into one value between 0 and 1. The instructor did not derive why that exact formula is used; for this lecture, the operational point was that values closer to 1 are better.

He also emphasized that these are representative measures rather than the only possible measures. The correct metric depends on what kinds of performance matter in the application.

---

## 15. Unequal costs of mistakes

### Written and visual content from the notes (PDF pages 3-4)

The notes state: **not all mistakes are equal**. The motivating application is a blood test used for breast-cancer diagnosis.

Page 4 contains a cost matrix analogous to the confusion matrix. Correct predictions have zero cost in the illustrative matrix. The two error types receive different costs, shown in the current notes as roughly:

- false positive: \(\$10{,}000\);
- false negative: \(\$1{,}000{,}000\).

The average-cost formula is

\[
\text{Average cost}
=
\frac{
TP\,C_{++}
+TN\,C_{--}
+FP\,C_{+-}
+FN\,C_{-+}
}{N_{\text{test}}}.
\]

The exact subscript convention depends on whether the first sign denotes the prediction or truth; the key idea is that each confusion-matrix outcome is multiplied by its assigned cost.

### Current-year spoken explanation

The instructor considered a classifier based on a blood test for breast-cancer diagnosis. The purpose is to obtain an inexpensive and relatively noninvasive first assessment. A positive result normally leads to further, more accurate testing; a negative result may send the patient home until a later screening.

A false negative is potentially catastrophic. The patient actually has cancer, but the test says that everything is fine. Treatment is delayed, the disease may progress, later treatment can be more expensive, and the hospital or software provider may face legal liability. After classroom suggestions, the instructor assigned an illustrative cost of about one million dollars.

A false positive is also harmful, but usually less severe. A patient without cancer is frightened and undergoes additional, possibly uncomfortable and expensive tests before the error is resolved. The class assigned an illustrative cost around ten thousand dollars.

The instructor stressed that the numbers were illustrative and application-dependent. The important point is that a model with high ordinary accuracy may still be unacceptable if it makes the expensive kind of error too often. A designer aware of these costs may deliberately prefer more false alarms if that sharply reduces missed cancers.

Average cost summarizes this by multiplying each outcome count by its consequence and dividing by the number of test cases.

---

## 16. Extending evaluation to multiclass classification

### Written and visual content from the notes (PDF page 4)

The page asks what happens when there are more than two classes. For three classes, it draws a \(3\times3\) confusion table. It indicates:

- percent accuracy remains applicable;
- the simple binary F1 setup is not used directly in the same form;
- average cost remains possible if a full \(3\times3\) cost matrix can be defined.

### Current-year spoken explanation

With three classes, the confusion matrix becomes \(3\times3\); with ten classes, it becomes \(10\times10\). The binary names TP, TN, FP, and FN no longer provide a convenient name for every cell.

Accuracy still works. Correct predictions lie on the main diagonal. Add the diagonal counts and divide by the total number of examples. Off-diagonal cells represent mistakes.

The binary F1 formula is not directly transferred in the simple form just introduced, because it relies on one positive class and the associated TP/FP/FN counts. The instructor did not introduce multiclass averaging variants in this lecture.

Cost-sensitive evaluation does generalize. If a cost can be assigned to every true-class/predicted-class combination, the full cost matrix can be combined with the corresponding confusion-matrix counts to compute average cost.

---

## 17. Iris demonstration: what KNN decision regions look like

### Written and visual relationship to the notes

The handwritten PDF does not contain the Python output shown in class. The closest written visual is the page-2 scatterplot illustrating labeled regions and nearest-neighbor reasoning. The current-year recording adds a substantial live software demonstration.

### Current-year spoken explanation

The instructor switched to a Python notebook and introduced the Iris dataset as a multiclass example:

- 150 flower examples;
- 3 species/classes;
- 50 examples in each class;
- 4 numerical measurements per flower.

He used scikit-learn and imported the nearest-neighbor tools. For visualization, he deliberately discarded the third and fourth features and retained only the first two, because the decision regions had to be drawn in two dimensions.

The notebook created a KNN classifier with equal voting weight for the selected neighbors. Much of the visible code prepared the plot rather than implementing the learning method itself.

### \(k=1\): each point can create its own region

With one nearest neighbor, every location in the plot is assigned the class of its single closest training flower. The colored background therefore shows a patchwork of regions. A lone blue point embedded among red points can create a small blue “island,” because it is the nearest example within that tiny region.

A student observed that with \(k=1\), every outlier receives its own zone. The instructor agreed and used this to illustrate sensitivity to noise or atypical examples.

### Increasing \(k\)

With \(k=3\), some tiny islands disappear because one isolated example loses the majority vote against its surrounding neighbors.

With \(k=9\), the regions change again. The instructor noted that increasing \(k\) does not guarantee a visually smooth or obviously sensible boundary; some unexpected regions can appear because of the particular distribution of the nine selected neighbors.

He then tried much larger values, including roughly 55 neighbors and values approaching the whole dataset. Surprisingly large values could still look superficially reasonable in parts of the plot, but the extreme case made the failure clear: when every training example votes, every location receives the same global majority prediction.

The conclusion was the same as the handwritten note: \(k\) must be explored rather than chosen automatically as either 1 or the dataset size.

---

## 18. Selecting \(k\) through repeated train-test experiments

### Written and visual content from the notes

Page 2 says that choosing \(k\) requires trial and error, while page 3 supplies the train-test evaluation framework. The current-year notebook combines these two ideas.

### Current-year spoken explanation

The instructor used scikit-learn to split the Iris data randomly into approximately 66% training and 33% testing. The classifier was created from the training data and accuracy was calculated on the test data.

Because the split is random, rerunning the experiment changes which flowers appear in the test set and therefore changes the measured accuracy. For \(k=1\), the live runs produced values around:

- 76%;
- 72%;
- 78%.

The instructor said that one should repeat the process several times - perhaps ten - to obtain a more stable sense of expected performance.

He then tried \(k=5\). Example runs produced approximately:

- 82%;
- 74%;
- 80%.

These individual numbers were not treated as definitive proof, but they suggested that \(k=5\) might perform somewhat better than \(k=1\) on this particular setup.

The operational hyperparameter-selection rule demonstrated here was:

1. choose a candidate \(k\);
2. randomly split the data;
3. train on the training subset;
4. evaluate on the test subset;
5. repeat to observe variability;
6. compare candidate values using their repeated performance.

The instructor described this as the basic trial-and-error idea and said that students would work with similar code in homework.

---

## 19. Demonstration of other classifier families

### Written and visual relationship to the notes

PDF page 1 says that a prediction model may be a mathematical function, a search-based algorithm, or a rule-based algorithm. The current-year notebook visualized how several such families partition feature space.

### Current-year spoken explanation

The instructor displayed three artificial two-dimensional datasets:

- two interlocking moon-like classes;
- an inner circle and an outer circle;
- two classes separated approximately into left and right regions.

For each dataset, the plots showed classifier decision regions. Darker regions represented stronger confidence in one class, while lighter colors indicated mixed neighborhoods or lower confidence.

The KNN boundary could be highly irregular because it followed local examples. A linear classifier drew a straight boundary, assigning one side to one class and the other side to the other. Decision-tree classifiers produced partitions built from horizontal and vertical cuts, creating rectangle-like regions. Neural-network examples produced more flexible boundaries that could follow the moons or circular structure more closely.

The instructor said that the course would eventually cover representatives of these major families and that the more flexible middle examples were visually appealing and potentially powerful.

### Training cost versus prediction cost

A student connected the demonstration to the earlier statement that KNN is expensive. The instructor distinguished two stages:

- **model construction/training:** basic KNN does almost nothing beyond storing the dataset;
- **prediction:** KNN must search or scan the stored examples for every new query.

Other algorithms may spend more effort learning a formula or compact model, but once learned they can often make predictions much faster than scanning the entire training set.

---

## 20. Is KNN a good supervised-learning algorithm?

### Written and visual content from the notes (PDF page 4)

The page asks: **Is KNN a good supervised-learning algorithm?**

Its answer is **yes, under ideal conditions**. The ideal condition is a large amount of data creating very dense neighborhoods. With many extremely close neighbors, the local class distribution can be estimated accurately.

The notes then warn that real machine-learning datasets rarely provide such neighborhoods.

### Current-year spoken explanation

The instructor said KNN can be very good under ideal conditions. The key requirement is not merely “a large dataset” in the abstract but dense local neighborhoods.

He used a geographic analogy:

- living in Manhattan means many people are extremely close;
- living in the middle of Pennsylvania may mean the closest person is still far away.

KNN needs **near** neighbors, not merely the least-far example. If the closest neighbor is 100 miles away, guilt by association becomes a weak basis for prediction.

In an ideally dense neighborhood, many examples occupy almost the same location in feature space. One can count the class frequencies there and estimate

\[
P(y\mid x),
\]

the probability of each label given the features. If a million nearly identical examples surround the query, their local proportions can provide a very accurate probability estimate.

Unfortunately, typical machine-learning datasets rarely have neighborhoods this dense, even when the dataset contains a million observations.

---

## 21. The curse of dimensionality: neighborhoods disappear

### Written and visual content from the notes (PDF page 4)

The handwritten page states:

> As the number of features grows, the neighborhood disappears.

It labels this phenomenon the **curse of dimensionality**.

A geometric construction considers \(N\) examples uniformly distributed in a unit space. Around a point in the center, define a neighborhood whose side length is one-half of the full space.

### One dimension

The neighborhood occupies one-half of the interval, so the expected number of neighbors is

\[
N\left(\frac12\right).
\]

### Two dimensions

A centered square with side length \(1/2\) occupies area

\[
\left(\frac12\right)^2=\frac14,
\]

so the expected number of points is

\[
N\left(\frac12\right)^2=\frac{N}{4}.
\]

### Three dimensions

A centered cube with side length \(1/2\) occupies volume

\[
\left(\frac12\right)^3=\frac18,
\]

so the expected number is

\[
N\left(\frac12\right)^3=\frac{N}{8}.
\]

### \(M\) dimensions

The general expression is

\[
N\left(\frac12\right)^M.
\]

The notes use

\[
N=1{,}000{,}000=10^6,
\qquad M=100.
\]

Because \(2^{100}\) is on the order of \(10^{30}\),

\[
10^6\left(\frac12\right)^{100}
\approx 10^{-24}.
\]

Thus, even a million examples produce essentially zero expected neighbors in this very large neighborhood.

The page also states the distance-concentration result

\[
\frac{\text{distance to nearest neighbor}}
{\text{distance to farthest neighbor}}
\to 1
\quad\text{as }M\to\infty.
\]

### Current-year spoken explanation

The instructor called the curse of dimensionality an enemy of machine learning generally and an especially serious problem for KNN. Human geometric intuition comes from two- and three-dimensional space. In ordinary geography, Philadelphia contains many nearby people, while Australia is clearly far away. We intuitively separate near and far.

That intuition breaks down in a space with 50, 100, or more features. As \(M\) grows, local neighborhoods become sparse even when \(N\) is very large. In the instructor's phrasing, the neighborhood “disappears.”

He also stated the more surprising consequence: the closest and farthest observations become nearly equally distant relative to the overall scale. In a 100-dimensional world, there may be no genuinely nearby example, and all examples may lie at roughly the same distance.

### Geometric classroom derivation

The instructor drew a unit square containing \(N\) points and placed a person or query at the center. He defined a generous square neighborhood with side length 0.5. In two dimensions, it contains one quarter of the total area and therefore roughly \(N/4\) points.

He then drew a unit cube and a smaller cube of side 0.5. Its volume is one eighth, giving \(N/8\) expected points. Looking backward to one dimension gives \(N/2\), making the pattern clear:

\[
N/2^M.
\]

For one million observations and 100 features, this becomes approximately \(10^{-24}\), effectively no neighbor at all. The instructor said that even expanding the neighborhood to nearly the full width of the space may still fail to include a genuinely local example.

He ended with the image that, in high dimensions, points tend to lie on a distant shell around the query rather than filling a familiar local neighborhood.

---

