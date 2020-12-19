# Classifying NBA Player Sentiment Through Player Interviews
## Skyler Hallinan, Arthur Beyer
### CSE 490G1 Final Project

Write a 3-4 sentence abstract. It should introduce the problem and your approach. You may also want some numbers like 35 mAP and 78% accuracy. You can use this example README for your project, you can use a different ordering for your website, or you can make a totally different website altogether!

VIDEO GOES HERE (probably): Record a 2-3 minute long video presenting your work. One option - take all your figures/example images/charts that you made for your website and put them in a slide deck, then record the video over zoom or some other recording platform (screen record using Quicktime on Mac OS works well). The video doesn't have to be particularly well produced or anything.

[![Video](https://img.youtube.com/vi/u0XEFaEItrQ/0.jpg)](https://www.youtube.com/watch?v=u0XEFaEItrQ)


## Introduction

In the National Basketball Association (NBA), teamwork is essential. However, chemistry issues between teammates often plague teams, leading to issues on and off the court, potentially leading to losses, social media bouts, and players being traded to other teams. It is therefore essential for teams to identify and remediate public and private discontent expressed by its players to prevent implosions.
 
One example of such an implosion was on the Minnesota Timberwolves in 2019. After a particularly rough stretch former player Jimmy Butler famously challenged his teammates, saying “You f——ng need me! You can’t win without me!” This caused a rift and days later he was traded to another team. Although chemistry issues sometimes manifest in these explicit and obvious forms, other problems may be more subtle. Before he was traded, former Cleveland Cavalier Isaiah Thomas remarked to reporters, “That’s what this team has been this whole season: inconsistent.” Although not as direct as Butler’s statement, this was a similarly negative statement to Butler’s, showing his discontent with the team. 
    
To target and dispel these issues associated with player unhappiness, in this work we seek to build an automatic classifier that can quickly assess the sentiment given a particular response or interview from a player. Although there exists one dataset of NBA player interviews, almost all of the data comes from formal press conferences which often involve other players, coaches, and/or a more scrutinous media; this may restrict the extent to which players speak freely and may not be helpful to annotate on sentiment, as we anticipate a homogenous set of somewhat uncontroversial responses.

Instead, we train our model on informal postgame, pregame, and post-practice interviews, which we denote Informal Game and Practice Interviews (IGPIs). These interviews are semi-structured, question-and-answer style discussions, where players freely answer questions from reporters without the influence of coaches or teammates. As such, IGPIs may reveal important insights into a player’s relationships with their team and coaching staff. 

However, there is a lack of public annotated IGPIs. To remediate this, we present a novel
corpus of about 150 IGPIs ranging from one to ten minutes each, from 33 players over the last three years, each transcribed verbatim from publicly available video interviews online on YouTube and on NBA.com. We also provide metadata related to these interviews, such as the date, the player name, and what team they were on. Finally, we also provide sentiment annotations for each interview for our classification task, using a scale from 0-3.

With the new corpus, we train several sentiment classification models with a variety of LSTM architectures on both character and word level embeddings and demonstrate the advantages and disadvantages. Finally, we show a demonstration of our system in use.


## Related Work

Other people are out there doing things. What did they do? Was it good? Was it bad? Talk about it here.

## Approach

One of the biggest parts of this project was the annotation. We spent a collective roughly 40 hours meticulously annotating the interviews verbatim. We even included words like “Um” and “Uh” and included all punctuation, to the best of our ability. (We estimate each interview is 4 minutes long. And for each interview minute, it takes about 4 minutes to annotate, so each interview roughly takes 16 minutes to annotate. We have 143 interviews, so that’s 16*143=2288 minutes converted to hours is about 40 hours).

Once we had the interviews annotated, we next had to annotate the sentiment for each interview. We chose a sentiment score from 0 to 3 for each player where we perceived their attitude to be:
* 0: Extremely unhappy/discontent
* 1: Somewhat unhappy/discontent
* 2: Somewhat happy/content
* 3: Extremely happy/content
This took about 2 minutes per interview so roughly 4 hours total

Once we had all the data we were going to use for this project, we started working on finding the best model that could fit our data. We considered parsing our data character-by-character and word-by-word. We also considered using a bag-of-words approach but thought that sequence based models would perform better than simply counting appearances of words. For this reason, we relied on LSTM and Bidirectional LSTM models for both character-by-character and word-by-word parsing. We also tried multi-layer LSTM models but found that it offered little benefit, if any to the accuracy. Since the interviews varied significantly in length, we briefly explored sub-sampling but found that it made the accuracy way worse, little better than guessing. We found that character-by-character parsing performed slightly better after tuning hyper-parameters.

## Results

How did you evaluate your approach? How well did you do? What are you comparing to? Maybe you want ablation studies or comparisons of different methods.

You may want some qualitative results and quantitative results. Example images/text/whatever are good. Charts are also good. Maybe loss curves or AUC charts. Whatever makes sense for your evaluation.

### Character Level Results with Full Sequence Padding
![epochsnodesBaseLstm](https://user-images.githubusercontent.com/28735634/102581159-d6627600-40b4-11eb-83b4-b073133c8dda.png)
![epochsnodes2](https://user-images.githubusercontent.com/28735634/102577550-19b8e680-40ad-11eb-9a61-43e7fc0fcc93.png)
![twodropoutbidirectional](https://user-images.githubusercontent.com/28735634/102665497-c2148c80-4139-11eb-9334-c298002e6d8e.png)
![truncationbase](https://user-images.githubusercontent.com/28735634/102665498-c345b980-4139-11eb-8bab-81097551a755.png)
![dropoutnodes](https://user-images.githubusercontent.com/28735634/102665499-c345b980-4139-11eb-90a2-88ec474b9783.png)
![epochsBatchSizeBase](https://user-images.githubusercontent.com/28735634/102665671-13248080-413a-11eb-8804-141df86c8956.png)

###
![evls_nodes_multi](https://user-images.githubusercontent.com/47925992/102675464-55f05380-414e-11eb-8548-acf881bbf153.png)
![evls_nodes_multi_bi](https://user-images.githubusercontent.com/47925992/102675465-5688ea00-414e-11eb-8857-9b6237b122e3.png)
![bsize_nodes](https://user-images.githubusercontent.com/47925992/102675466-5688ea00-414e-11eb-8aab-282f2d900b5c.png)
![evls_nodes](https://user-images.githubusercontent.com/47925992/102675469-57218080-414e-11eb-9fe3-c4458c226e4f.png)
![evls_nodes_bidirectional](https://user-images.githubusercontent.com/47925992/102675470-57218080-414e-11eb-8479-e1064354588f.png)
![bsize_nodes_bidirectional](https://user-images.githubusercontent.com/47925992/102675471-57218080-414e-11eb-979c-339346806614.png)

### Overall results
![best_char](https://user-images.githubusercontent.com/28735634/102678550-32cda000-415e-11eb-81b4-4ae04ec08066.png)
![best_Word](https://user-images.githubusercontent.com/28735634/102678553-33fecd00-415e-11eb-86dc-f5e16fb7949a.png)

## Discussion

Before this project, we knew that machine learning took a lot of data to be effective but we did not know how much work and time needs to go into generating the data. Even after over 40 hours combined of work, the we still feel like we had a tiny dataset. Since we had so little data, we certainly were not expecting super high accuracies. With something as complicated as natural language processing, we were expecting abysmal results with only 143 data points. But we actually got a relatively high accuracy all things considered: hovering around 35-40% which is significantly better than guessing which should be around 25%. If we had more data, we expect our accuracy to increase significantly.

In the future, we want to use something like this to predict trades before they occur: if a player is consistently receiving low sentiment marks in his interviews, it likely means he wants out of a team and a trade is imminent.

Additionally, since the size/length of the interviews varied so much (from just a few sentences to paragraphs in length), we also want to consider more deeply sub-sampling because we believe it could be very promising if we had more data.

