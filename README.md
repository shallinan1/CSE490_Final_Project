# Classifying NBA Player Sentiment with a New Corpus
## Skyler Hallinan, Arthur Beyer
### CSE 490G1 Final Project

## Abstract

In this project we analyze informal NBA player interviews in an effort to estimate player sentiment. We use Bidirectional and Base LSTM models to process the interviews and end up with a peak accuracy of around 35-40%. With more data, we expect this accuracy to increase drastically.

## Video

[![Video](https://user-images.githubusercontent.com/28735634/102682177-63bccd80-417c-11eb-9667-0600366757ab.png)](https://www.youtube.com/watch?v=MgP0RqchFjE)
(click the image or [this link](https://www.youtube.com/watch?v=MgP0RqchFjE) to view the video)
## Introduction

In the National Basketball Association (NBA), teamwork is essential. However, chemistry issues between teammates often plague teams, leading to issues on and off the court, potentially leading to losses, social media bouts, and players being traded to other teams. It is therefore essential for teams to identify and remediate public and private discontent expressed by its players to prevent implosions.
 
One example of such an implosion was on the Minnesota Timberwolves in 2019. After a particularly rough stretch former player Jimmy Butler famously challenged his teammates, saying “You f——ng need me! You can’t win without me!” This caused a rift and days later he was traded to another team. Although chemistry issues sometimes manifest in these explicit and obvious forms, other problems may be more subtle. Before he was traded, former Cleveland Cavalier Isaiah Thomas remarked to reporters, “That’s what this team has been this whole season: inconsistent.” Although not as direct as Butler’s statement, this was a similarly negative statement to Butler’s, showing his discontent with the team. 
    
To target and dispel these issues associated with player unhappiness, in this work we seek to build an automatic classifier that can quickly assess the sentiment given a particular response or interview from a player. Although there exists one dataset of NBA player interviews, almost all of the data comes from formal press conferences which often involve other players, coaches, and/or a more scrutinous media; this may restrict the extent to which players speak freely and may not be helpful to annotate on sentiment, as we anticipate a homogenous set of somewhat uncontroversial responses.

Instead, we train our model on informal postgame, pregame, and post-practice interviews, which we denote Informal Game and Practice Interviews (IGPIs). These interviews are semi-structured, question-and-answer style discussions, where players freely answer questions from reporters without the influence of coaches or teammates. As such, IGPIs may reveal important insights into a player’s relationships with their team and coaching staff. 

However, there is a lack of public annotated IGPIs. To remediate this, we present a novel
corpus of about 150 IGPIs ranging from one to ten minutes each, from 33 players over the last three years, each transcribed verbatim from publicly available video interviews online on YouTube and on NBA.com. We also provide metadata related to these interviews, such as the date, the player name, and what team they were on. Finally, we also provide sentiment annotations for each interview for our classification task, using a scale from 0-3.

With the new corpus, we train several sentiment classification models with a variety of LSTM architectures on both character and word level embeddings and demonstrate the advantages and disadvantages. Finally, we show a demonstration of our system in use.


## Related Work

We found a study that predicted in-game performances based on previous performance metrics in addition to player interviews before the game. This study used neural networks to combine text data and numerical performance data into one model. They tried many different approaches including Bag of Words with TFIDF, LSTM, BiLSTM, DNN, CNN, and a BERT model. Their accuracy exceeded that of analyzing performance metrics alone. The biggest difference between their study and ours is that they were able to draw from over 5,000 player interviews across 10 years whereas we were limited to however much we could closely transcribe in a couple weeks: 143 interviews.
https://arxiv.org/abs/1910.11292


## Approach

One of the biggest parts of this project was the annotation. We spent a collective roughly 40 hours meticulously annotating the interviews verbatim. We even included words like “Um” and “Uh” and included all punctuation, to the best of our ability. (We estimate each interview is 4 minutes long. And for each interview minute, it takes about 4 minutes to annotate, so each interview roughly takes 16 minutes to annotate. We have 143 interviews, so that’s 16*143=2288 minutes converted to hours is about 40 hours).

Once we had the interviews annotated, we next had to annotate the sentiment for each interview. We chose a sentiment score from 0 to 3 for each player where we perceived their attitude to be:
* 0: Extremely unhappy/discontent
* 1: Somewhat unhappy/discontent
* 2: Somewhat happy/content
* 3: Extremely happy/content
This took about 2 minutes per interview so roughly 4 hours total

Once we had all the data preprocessed, we started working on finding the best model that could fit our data. We considered parsing our data character-by-character and word-by-word. We also considered using a bag-of-words approach but thought that sequence based models would perform better than simply counting appearances of words. For this reason,we relied on LSTM and Bidirectional LSTM models for both character-by-character and word-by-word parsing. A Bidirectional LSTM is an LSTM that takes into account future words into its prediction. We also tried multi-layer LSTM models because we thought that a more complicated model could perform better than the simpler models we tried earlier but found that it offered little benefit, if any to the accuracy. This is probably because we were already overfitting with the simpler model based on the limited amount of data we had. For the word based sequencing, we tried varying the word embedding length which gave interesting results. Since the interviews varied significantly in length, we briefly explored sub-sampling but found that it made the accuracy way worse, little better than guessing. We found that character-by-character parsing performed slightly better after tuning hyper-parameters. To generate the data, we took averages of 3 runs for each of the settings of the hyper-parameters to ensure that we were not getting bogus results.

![132012084_671673920184354_8767085777290283440_n](https://user-images.githubusercontent.com/47925992/102682350-c6629900-417d-11eb-8fca-202a603ccbcf.png)
Character-Based Model

![131986515_427335658675797_4274136360118186489_n](https://user-images.githubusercontent.com/47925992/102682351-c6629900-417d-11eb-951b-7a93af0b525c.png)
Word-Based Model

## Results

We evaluated our results using test accuracy: how many in the test set our model got correct divided by how many total are in the test set. Even though our accuracy in these graphs seems low, we believe it is actually high considering how little data we had (see Discussion section).

### Character Level Results with Full Sequence Padding
![epochsnodesBaseLstm](https://user-images.githubusercontent.com/28735634/102581159-d6627600-40b4-11eb-83b4-b073133c8dda.png)
This one had a batch size of 16 and a padding size of 500. Note the accuracy for 50 nodes at 5 epochs is high but it decreases as the number of epochs increases. This could imply that we are overfitting at 50 nodes.

![epochsnodes2](https://user-images.githubusercontent.com/28735634/102577550-19b8e680-40ad-11eb-9a61-43e7fc0fcc93.png)
This model performed significantly worse than without dropout.
This one had a batch size of 16 and a padding size of 500. The accuracy semed to go down then back up throughout the epochs.

![twodropoutbidirectional](https://user-images.githubusercontent.com/28735634/102665497-c2148c80-4139-11eb-9334-c298002e6d8e.png)
This model performed significantly worse than without dropout.

![truncationbase](https://user-images.githubusercontent.com/28735634/102665498-c345b980-4139-11eb-8bab-81097551a755.png)
This model also performed significantly worse than without truncation.

![dropoutnodes](https://user-images.githubusercontent.com/28735634/102665499-c345b980-4139-11eb-90a2-88ec474b9783.png)
Seems like dropout is not the move

![epochsBatchSizeBase](https://user-images.githubusercontent.com/28735634/102665671-13248080-413a-11eb-8804-141df86c8956.png)
The batch size seems not to affect the accuracy very much.

### Word Level Results with Full Sequence Padding
![evls_nodes_multi](https://user-images.githubusercontent.com/47925992/102675464-55f05380-414e-11eb-8548-acf881bbf153.png)
For the word tests, we kept the epochs constant at 10 and instead opted to change the embedding vector length and the batch size.

![evls_nodes_multi_bi](https://user-images.githubusercontent.com/47925992/102675465-5688ea00-414e-11eb-8857-9b6237b122e3.png)
Bidirectional does not seem to significantly improve performance over a base LSTM.

![bsize_nodes](https://user-images.githubusercontent.com/47925992/102675466-5688ea00-414e-11eb-8aab-282f2d900b5c.png)
The batch size did seem to change the accuracy of the model but oddly, a batch size of 16 performed well but a batch size of 64 performed arguably just as well, just with a higher variance.

![evls_nodes](https://user-images.githubusercontent.com/47925992/102675469-57218080-414e-11eb-9fe3-c4458c226e4f.png)
Varying the embedding length gave interesting results. It seems like length 32 with 15 LSTM nodes had the highest accuracy here.

![evls_nodes_bidirectional](https://user-images.githubusercontent.com/47925992/102675470-57218080-414e-11eb-8479-e1064354588f.png)
The bidirectional LSTM had a lower variance than the base LSTM but roughly the same average accuracy across all nodes.

![bsize_nodes_bidirectional](https://user-images.githubusercontent.com/47925992/102675471-57218080-414e-11eb-979c-339346806614.png)
The batch size seems to have a medium point where the variance explodes then gets smaller on both ends of the middle.


### Overall results
![best_char](https://user-images.githubusercontent.com/28735634/102678550-32cda000-415e-11eb-81b4-4ae04ec08066.png)
![best_Word](https://user-images.githubusercontent.com/28735634/102678553-33fecd00-415e-11eb-86dc-f5e16fb7949a.png)

## Discussion

Before this project, we knew that machine learning took a lot of data to be effective but we did not know how much work and time needs to go into generating the data. Even after over 40 hours combined of work, the we still feel like we had a tiny dataset. Since we had so little data, we certainly were not expecting super high accuracies. With something as complicated as natural language processing, we were expecting abysmal results with only 143 data points. But we actually got a relatively high accuracy all things considered: hovering around 35-40% which is significantly better than guessing which should be around 25%. If we had more data, we expect our accuracy to increase significantly.

In the future, we want to use something like this to predict trades before they occur: if a player is consistently receiving low sentiment marks in his interviews, it likely means he wants out of a team and a trade is imminent.

Additionally, since the size/length of the interviews varied so much (from just a few sentences to paragraphs in length), we also want to consider more deeply sub-sampling because we believe it could be very promising if we had more data.

We could also look into pre-trained word embeddings like glove which show similarities between words (for example, if there is a vector that corresponds to the word: “king”, we could subtract the vector for “man” and add the vector for “woman” and get a vector that looks like “queen”. This could be useful when a new word shows up in a new interview that we have not seen before but is very similar to another word: (think “magnificent” and “great” as an example).

We could also consider doing more complex annotations other than a simple 0-3 number. For example, if the interview is "We fought well today and ultimately we got the win together as a team", the annotation could be "This player won and likes his team".

## Demo
![Demo](https://user-images.githubusercontent.com/28735634/102678608-6f999700-415e-11eb-8b0c-35262254834c.gif)

