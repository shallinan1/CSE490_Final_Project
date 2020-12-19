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
Our simple annotated dataset is available in the Github, but does not contain the described metadata (name, team, date, etc) for storage reasons. Please contact us directly if you would like to use the full dataset.

We evaluated our results using test accuracy: how many in the test set our model got correct divided by how many total are in the test set; we used an 80/20 split for our train and test data for this project. Even though our accuracy in these graphs seems low, we believe it is actually high considering how little data we had (see discussion section).

Below are first our character level sequence model results:

### Character Level Experiment Results with Full Sequence Padding
![epochsBatchSizeBase](https://user-images.githubusercontent.com/28735634/102665671-13248080-413a-11eb-8804-141df86c8956.png)
We first tested batch size vs number of nodes in one LSTM using 10 epochs. However, the batch size seemed to not to affect the accuracy very much, but 16 was the best. We had a max of about 0.375 with 100 noes and a batch size of 16. We used 16 as our batch size for all future experiments with this character-based model. Next, we tested the number of epochs in our model vs the number of nodes in our LSTM:

![epochsnodesBaseLstm](https://user-images.githubusercontent.com/28735634/102581159-d6627600-40b4-11eb-83b4-b073133c8dda.png)
Using a batch size of 16 and one LSTM, we varied number of nodes in our LSTM and the number of epochs. Note the accuracy for 50 nodes at 5 epochs is high at 0.43 but it decreases as the number of epochs increases. This could imply that we are overfitting with too many epochs. Overall, the max average classification accuracy here was with 50 LSTM nodes and an accuracy of 0.43.

![epochsnodes2](https://user-images.githubusercontent.com/28735634/102577550-19b8e680-40ad-11eb-9a61-43e7fc0fcc93.png)
We repeated the same experiment as directly above, but using a bidirectional LSTM. Across all node numbers, the accuracy semed to go down then back up throughout the epochs. Overall, we had max accuracies of 0.43 with 25 nodes and 20 epochs and 100 nodes and 3 epochs.


![dropoutnodes](https://user-images.githubusercontent.com/28735634/102665499-c345b980-4139-11eb-90a2-88ec474b9783.png)
Next, we tried adding dropout layers before and after to our base LSTM to see how it would affect the accuracy. The accuracy steadily went down for all LSTMs when we increased the dropout proportion, and the accuracy even with low dropout was much lower than 0.43, our previous best from other models. Dropout did not help us much here.

![twodropoutbidirectional](https://user-images.githubusercontent.com/28735634/102665497-c2148c80-4139-11eb-9334-c298002e6d8e.png)
Finally, we repeated the above but with the bidirectional LSTM. Once again, this did not help much and our accuracy plumetted to almost 0.25 for all models, showing that dropout was not helping at all.

Overall, our character-level model had a best case average model accuracy on the test set of about 0.43 (proportion) and a best accuracy of about 0.51 for a single model. Batch size of 16 worked best, and dropout didn't help. LSTM and bidirectional LSTM were equally qualified for our classification task.

Next, we moved on to the word sequence experiments.

### Word Level Experiment Results with Full Sequence Padding
![bsize_nodes](https://user-images.githubusercontent.com/47925992/102675466-5688ea00-414e-11eb-8aab-282f2d900b5c.png)
For our word sequence experiments, we first looked at the effects of batch size on our data, as we also varied number of nodes in our single LSTM. For all experiments including this one, we used 10 epochs. Batch size did seem to change the accuracy of the model but oddly, a batch size of 16 performed well but a batch size of 64 performed arguably just as well, just with a higher variance. We used a batch size of 16 for the rest of our experiments. Note our max accuracy was about 0.36 here with 5 LSTM nodes. F

![bsize_nodes_bidirectional](https://user-images.githubusercontent.com/47925992/102675471-57218080-414e-11eb-979c-339346806614.png)
We repeated the above experiment but with a bidrectional LSTM. Differences across batch size were pretty negligible. Here our max accuracy was a batch size of 8, 35 nodes in our LSTM, and an accuracy of 0.37. We note the batch size seems to have a medium point where the variance explodes then gets smaller on both ends of the middle. Overall, even with these results we used 16 as our batch size for our experiments.

![evls_nodes](https://user-images.githubusercontent.com/47925992/102675469-57218080-414e-11eb-9fe3-c4458c226e4f.png)
Next, we tried varying the embedding vector length and changing the number of nodes in a single LSTM. Increasing vector embedding potentially allows for more complex representations of words. This gave interesting results. It seems like length 32 with 15 LSTM nodes had the highest accuracy here, with an average accuracy of about 0.36.

![evls_nodes_bidirectional](https://user-images.githubusercontent.com/47925992/102675470-57218080-414e-11eb-8479-e1064354588f.png)
We repeated the above experiment but with bidirectional LSTM. It had a lower variance than the base LSTM but a slightly lower average accuracy across all nodes, peaking at about 0.35 with 5 nodes an embedding size of 64. Therefore, the results were slightly worse than with the base LSTM.

![evls_nodes_multi](https://user-images.githubusercontent.com/47925992/102675464-55f05380-414e-11eb-8548-acf881bbf153.png)
Finally, we tested a multi LSTM network with two LSTM layers and multiple embedding vector lengths. Note that in our first LSTM layer we had 50 nodes while we varied only the number of nodes in the second LSTM. We achieved a max average accuracy of about 0.39 with 35 nodes in our second LSTM and an embedding vector length of 32. This is higher than any other max average accuracy in the other networks we tried. Therefore, this multi LSTM network did pretty well.

![evls_nodes_multi_bi](https://user-images.githubusercontent.com/47925992/102675465-5688ea00-414e-11eb-8857-9b6237b122e3.png)
Finally, we repeated the above with a stacked bidirectional network, changing the number of nodes in only the second layer (the first LSTM was fixed at 50 nodes). Although the results on average were slightly worse than the multi-layer base LSTM network, we still had a model with good accuracy: with 25 nodes and an embedding length of 16, the model had an accuracy of 0.39, tying it for best with the previous network.

Overall, the word based models were worse than the character-based models, peaking with an accuracy of 0.39 in both stacked LSTM networks (base and bidirectional). The stacked network architechture helped the performance and did not delay training too much, so it is probably worth it. However, the character-based model was better so we would recommen using that instead.

Finally, we describe our overall results.
### Overall results
![best_char](https://user-images.githubusercontent.com/28735634/102678550-32cda000-415e-11eb-81b4-4ae04ec08066.png)

Overall, the character-models performed better than the word-based model with the bidirectional LSTM and without the LSTM, peaking with an average classification accuracy of about 0.43. The ideal number of LSTM nodes to use hear was about 25-100.

![best_Word](https://user-images.githubusercontent.com/28735634/102678553-33fecd00-415e-11eb-86dc-f5e16fb7949a.png)


The word-based models had lower accuracy and peaked at about 0.39. They seemed to perform best with a stacked bidirectional LSTM and 35 nodes in the second LSTM, and 50 nodes in the first LSTM, achieving the highest classification accuracy. Overall, the results were varied across network types, but all were worse than the character-based models.

## Discussion

Before this project, we knew that machine learning took a lot of data to be effective but we did not know how much work and time needs to go into generating the data. Even after over 40 hours combined of work, we still feel like we had a tiny dataset. Since we had so little data, we certainly were not expecting super high classification accuracies. With something as complicated as natural language processing, we were expecting abysmal results with only 143 data points. But we actually got a relatively high accuracy (with the character-based network) all things considered: hovering around 43% which is significantly better than guessing which should be around 25%. If we had more data, we expect our accuracy to increase significantly. We were surprised that the accuracy with words was almost 4% lower than with characters, but this may be an artifact of small sample size.

In the future, we might construct a system like this to predict trades before they occur: if a player is consistently receiving low sentiment marks in his interviews, it likely means he wants out of a team and a trade is imminent. This could be helpful in the betting scenes, were people often place prop bets on players being traded to different destinations. However, it would be most helpful for competitor teams to know when trades were able to happen, as they could prepare accordingly for the future.

Additionally, since the size/length of the interviews varied so much (from just a few sentences to paragraphs in length), we also want to consider more deeply sub-sampling because we believe it could be very promising if we had more data. Sub-sampling would consist of choosing smaller, random sequences from our preexisting dataset. This has the effect of homogeneizing the length of our sequences and reducing huge variability in length. It also increases our sample size because we can generate multiple interview snippets from a single interview. Using this method in the future might help us obtain more data, but we would have to be careful to make sure we're not overdoing it. 

We could also look into pre-trained word embeddings like GloVe which show similarities between words (for example, if there is a vector that corresponds to the word: “king”, we could subtract the vector for “man” and add the vector for “woman” and get a vector that looks like “queen”. This could be useful when a new word shows up in a new interview that we have not seen before but is very similar to another word: (think “magnificent” and “great” as an example), and would be good for our word-based models.

Finally, we could also consider doing more complex annotations other than a simple 0-3 number, as this sentiment score may be too simple and ultimately unhelpful, even if output at high accuracy. For example, if the interview is "We fought well today and ultimately we got the win together as a team", the annotation could be "This player won", "This player likes his team, and "This player is happy". Building a model to generate these inferences would be more helpful in accurately predicting players' emotions, but it would require much more annotation and an even larger corpus of data.

## Demo
![Demo](https://user-images.githubusercontent.com/28735634/102678608-6f999700-415e-11eb-8b0c-35262254834c.gif)

Here is a demo of our project that displays the sentiment associated with a given quote from a player. We test various inputs and show that it works relatively well, although it does not seem to output many extreme results ("Extremely happy" or "Extremely unhapy"). You can also run this on your own device by downloading the CSE490_Final_Project notebook, training the model, and feeding your own input.

