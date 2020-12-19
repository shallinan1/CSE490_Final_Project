# Classifying NBA Player Sentiment with a New Corpus
## Skyler Hallinan, Arthur Beyer
### CSE 490G1 Final Project, 12.18.2020
## Abstract
In this project we analyze informal NBA player interviews in an effort to estimate player sentiment. First, we construct a novel corpus consisting of informal player interviews and annotated sentiment on a scale of 0-3. Next, we build sentiment classification models using various LSTM networks, and end up with average accuracies of our best models around 43%, and peak accuracy (of a single model) of about 51%. With more data, we expect this accuracy to increase drastically.

## Video

[![Video](https://user-images.githubusercontent.com/28735634/102682177-63bccd80-417c-11eb-9667-0600366757ab.png)](https://www.youtube.com/watch?v=sd5wDg6kL_U)
(click the image or [this link](https://www.youtube.com/watch?v=sd5wDg6kL_U) to view the video)
## Introduction

In the National Basketball Association (NBA), teamwork is essential. However, chemistry issues between teammates often plague teams, leading to issues on and off the court, potentially leading to losses, social media bouts, and players being traded to other teams. It is therefore essential for teams to identify and remediate public and private discontent expressed by its players to prevent implosions.
 
One example of such an implosion was on the Minnesota Timberwolves in 2019. After a particularly rough stretch former player Jimmy Butler famously challenged his teammates, saying “You f——ng need me! You can’t win without me!” This caused a rift and days later he was traded to another team. Although chemistry issues sometimes manifest in these explicit and obvious forms, other problems may be more subtle. Before he was traded, former Cleveland Cavalier Isaiah Thomas remarked to reporters, “That’s what this team has been this whole season: inconsistent.” Although not as direct as Butler’s statement, this was a similarly negative statement to Butler’s, showing his discontent with the team. 

To target and dispel these issues associated with player unhappiness, in this work we seek to build an automatic classifier that can quickly assess the sentiment given a particular response or interview from a player. General sentiment classifiers exist, but they they have not been trained on any NBA player speech and are not therefore not well-suited for this setting. Therefore, we must build a classifier trained on an NBA-specific dataset. Although there exists one dataset of NBA player interviews, almost all of the data comes from formal press conferences which often involve other players, coaches, and/or a more scrutinous media; this may restrict the extent to which players speak freely and may not be helpful to annotate on sentiment, as we anticipate a homogenous set of somewhat uncontroversial responses.

Instead, we decide to train our model on informal postgame, pregame, and post-practice interviews, which we denote Informal Game and Practice Interviews (IGPIs). These interviews are semi-structured, question-and-answer style discussions, where players freely answer questions from reporters without the influence of coaches or teammates. As such, IGPIs may reveal important insights into a player’s relationships with their team and coaching staff. 

To remediate the lack of publicly available IGPIs online, we present a novel corpus of almost 150 IGPIs ranging from one to ten minutes each, from 33 players over the last three years, each transcribed verbatim from publicly available video interviews online on YouTube and on NBA.com. We also provide metadata related to these interviews, such as the date, the player name, and what team they were on. Finally, we also provide sentiment annotations for each interview for our classification task, using a scale from 0-3.

With the new corpus, we train several sentiment classification models with a variety of LSTM architectures on both character encodings and word level embeddings and demonstrate the advantages and disadvantages. Finally, we show a demonstration of our system in use.


## Related Work

A previous study, [Predicting In-game Actions from Interviewsof NBA Players](https://arxiv.org/pdf/1910.11292.pdf) predicted in-game performance statistics of NBA players, such as points, rebounds, and assists, based on both their previous performance metrics and their player interviews before the game. Note that the interview data they used is the aformentioned dataset from the introduction, consisting almost entirely of press conference interviews.
 
This study used neural networks to combine text data and numerical performance data into one model. They tried many different approaches including Bag of Words with TFIDF, as well as recurrent neural network models (LSTM, BiLSTM), transformer-based models (BERT), convolutional neural networks (CNN), and deep neural networks (DNN). Utilizing both interviews and previous peformance, the accuracy of the model exceeded that of using performance metrics alone, and their model-building was thus sucessful. The biggest difference between their study and ours is that they were able to draw from over 5,000 player interviews across 10 years whereas we were limited to however much we could closely transcribe in a couple weeks: 143 interviews. In addition, their data was biased heavily towards players who participated in the playoffs, as a large prorportion of the data came from playoff interviews, while our data equally represented players from all teams.


## Approach

The first step of the project was the data collection and annotation. Our original goal was to annotate and label 300 interviews, 10 from each of the 30 teams in the NBA, so we randomly selected 10 interviews from each team in the NBA happening from 2016-2019 that was available on YouTube or NBA.com. However, we overestimated our capability and ended up transcribing and annotating on average about 5 interviews from each team.

We spent a collective roughly 40 hours annotating the interviews verbatim. This means we included words like “Um” and “Uh”, pauses, stutters, and all punctuation to the best of our ability. We estimate each interview on average was 4 minutes long and for each interview minute, it took about 4 minutes to completely annotate. On average, each interview then took 16 minutes to annotate. We annotated 143 interviews, so that’s 16\*143 = 2288 minutes = about 40 hours. Note that for each interview, we also included the date of the interview, the player name, and what team they were on.

Once we had the interviews annotated, we next had to annotate the sentiment for each interview. We chose a sentiment score from 0 to 3 for each player where we perceived their attitude to be:
* 0: Extremely unhappy/discontent
* 1: Somewhat unhappy/discontent
* 2: Somewhat happy/content
* 3: Extremely happy/content
This took about 2 minutes per interview so roughly an additional 4 hours total.


Once we had all the data preprocessed, we started working on finding the best model to classify sentiment scores for text from NBA players. We considered parsing our data into character-by-character and word-by-word sequences. We also considered using a bag-of-words approach but thought that sequence based models would perform better than simply counting appearances of words. Thus, we decided to try to compare two models: a character sequence model, and a word sequence model. 

An LSTM is a recurrent neural network especially useful for sequence tasks that rely on time-series data, such as text sequences due to its cell memory unit. For this reason, we relied on using LSTM as our core layer when model building. We also experimented with Bidirectional LSTM, an LSTM that takes into account future words into its prediction, along with the past; we thought this might improve our classification accuracy. Finally, we experimented with using multi-layer LSTM models because we thought that a more complicated model could perform better than the simpler models we tried earlier. 

First, we started off with a character-level model. Our pipeline was as follows:

1. Convert each character in text sequences to integer embedding
2. Pad sequences so they all have same length
3. One-hot encode sentiment labels
4. Split up into train/test data
5. Train model and try different architectures and hyperparameters

For step 5, we tried varying the following to see how they could affect our classification accuracy: batch size, epochs, LSTM type, number of LSTMs, number of nodes in LSTMs, and dropout layers. We fed in our input, which was masked, went through a LSTM and then a linear layer. Our overall network was of the following form:

![132012084_671673920184354_8767085777290283440_n](https://user-images.githubusercontent.com/47925992/102682350-c6629900-417d-11eb-8fca-202a603ccbcf.png)

Note the dashed lines indicate multiple potential paths in architecture setup.

We had a masking layer after our input to ignore parts of the sequence that were just padding. From the above diagram, we extensively explored the following type of networks with different hyperparameters: LSTM, birectional LSTM, LSTM with dropout, birectional LSTM with dropout. Note that although our diagram includes multi-layer LSTM networks, we did not generate extensive data for that due to poor performance. After our LSTM, we had a single linear layer that mapped to a classification using a softmax activation function. For this task, we used categorical cross entropy loss, and Adam as our optimizer with a learning rate of 0.01.  We extensively looked at the effects of number of nodes in our LSTM in our results for this model.

Next, we started off with a word-level model. Our pipeline was as follows:

1. Preprocess data by removing punctuation
2. Convert each word in text sequences to integer embedding with fixed vocab size
3. Pad sequences so they all have same length
4. One-hot encode sentiment labels
5. Split up into train/test data
6. Train model (right)

For step 1, our preprocessing consisted of removing punctuation from the text. For step 5, we tried varying the following to see how they could affect our classification accuracy: batch size, type of LSTM, number of LSTMs, number of nodes in LSTMs, and word embedding vector size. For the word-based model, we chose to use a word embedding to increase the dimensionality of our words and improve their representaions; by mapping them to a higher dimensional space, we could capture complex relationships between similar and different words. We did not need to do this with the character model, since characters don't have as well-defined relationships with eachother as words do. This word embedding was masked, fed into the LSTM portion, then the linear layer. Our overall network was of the following form:

![131986515_427335658675797_4274136360118186489_n](https://user-images.githubusercontent.com/47925992/102682351-c6629900-417d-11eb-951b-7a93af0b525c.png)

Note the dashed lines indicate multiple potential paths in architecture setup.

Once again, we had a masking layer to ignore parts of the sequence that were just padding. From the above diagram, we extensively explored the following type of networks with different hyperparameters: LSTM, birectional LSTM, two LSTMs, two birectional LSTMs. After our LSTM(s), we had a single linear layer that mapped to a classification using a softmax activation function. For this task, we used categorical cross entropy loss, and Adam as our optimizer with a learning rate of 0.01. We extensively looked at the effects of embedding vector length in our results for this model.

## Results

We evaluated our results using test accuracy: how many in the test set our model got correct divided by how many total are in the test set; we used an 80/20 split for our train and test data for this project. Even though our accuracy in these graphs seems low, we believe it is actually high considering how little data we had (see discussion section).

Below are first our character level sequence model results:

### Character Level Results with Full Sequence Padding
![epochsBatchSizeBase](https://user-images.githubusercontent.com/28735634/102665671-13248080-413a-11eb-8804-141df86c8956.png)
We first tested batch size vs number of nodes in one LSTM using 10 epochs. However, the batch size seemed to not to affect the accuracy very much, but 16 was the best. We had a max of about 0.375 with 100 noes and a batch size of 16. We used 16 as our batch size for all future experiments with this character-based model. Next, we tested the number of epochs in our model vs the number of nodes in our LSTM:

![epochsnodesBaseLstm](https://user-images.githubusercontent.com/28735634/102581159-d6627600-40b4-11eb-83b4-b073133c8dda.png)
Using a batch size of 16 and one LSTM, we varied number of nodes in our LSTM and the number of epochs. Note the accuracy for 50 nodes at 5 epochs is high at 0.43 but it decreases as the number of epochs increases. This could imply that we are overfitting with too many epochs. Overall, the max average classification accuracy here was with 50 LSTM nodes and an accuracy of 0.43.

![epochsnodes2](https://user-images.githubusercontent.com/28735634/102577550-19b8e680-40ad-11eb-9a61-43e7fc0fcc93.png)
This model performed significantly worse than without dropout.
This one had a batch size of 16 and a padding size of 500. The accuracy semed to go down then back up throughout the epochs.

![twodropoutbidirectional](https://user-images.githubusercontent.com/28735634/102665497-c2148c80-4139-11eb-9334-c298002e6d8e.png)
This model performed significantly worse than without dropout.

![dropoutnodes](https://user-images.githubusercontent.com/28735634/102665499-c345b980-4139-11eb-90a2-88ec474b9783.png)
Seems like dropout is not the move


![truncationbase](https://user-images.githubusercontent.com/28735634/102665498-c345b980-4139-11eb-8bab-81097551a755.png)
This model also performed significantly worse than without truncation.

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

Before this project, we knew that machine learning took a lot of data to be effective but we did not know how much work and time needs to go into generating the data. Even after over 40 hours combined of work, the we still feel like we had a tiny dataset. Since we had so little data, we certainly were not expecting super high accuracies. With something as complicated as natural language processing, we were expecting abysmal results with only 143 data points. But we actually got a relatively high accuracy all things considered: hovering around 43% which is significantly better than guessing which should be around 25%. If we had more data, we expect our accuracy to increase significantly.

In the future, we want to use something like this to predict trades before they occur: if a player is consistently receiving low sentiment marks in his interviews, it likely means he wants out of a team and a trade is imminent.

Additionally, since the size/length of the interviews varied so much (from just a few sentences to paragraphs in length), we also want to consider more deeply sub-sampling because we believe it could be very promising if we had more data.

We could also look into pre-trained word embeddings like GlOve which show similarities between words (for example, if there is a vector that corresponds to the word: “king”, we could subtract the vector for “man” and add the vector for “woman” and get a vector that looks like “queen”. This could be useful when a new word shows up in a new interview that we have not seen before but is very similar to another word: (think “magnificent” and “great” as an example). 

Finally, we could also consider doing more complex annotations other than a simple 0-3 number, as this sentiment score may be too simple and ultimately unhelpful. For example, if the interview is "We fought well today and ultimately we got the win together as a team", the annotation could be "This player won", "This player likes his team, and "This player is happy". Building a model to generate these inferences would be more helpful in accurately predicting players' emotions, but it would require much more annotation and an even larger corpus of data.

## Demo
![Demo](https://user-images.githubusercontent.com/28735634/102678608-6f999700-415e-11eb-8b0c-35262254834c.gif)

Here is a demo of our project that displays the sentiment associated with a given quote from a player. We test various inputs and show that it works relatively well, although it does not seem to output many extreme results ("Extremely happy" or "Extremely unhapy"). You can also run this on your own device by downloading the CSE490_Final_Project notebook, training the model, and feeding your own input.

