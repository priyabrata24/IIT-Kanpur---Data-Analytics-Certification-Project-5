library(randomForest)
library(ggplot2)
library(dplyr)
library(tree)
library(cvTools)


setwd("G:\\Data Science\\Course Project\\Banking\\Dataset")

bank_train <- read.csv("bank-full_train.csv",stringsAsFactors = F)

bank_test <- read.csv("bank-full_test.csv",stringsAsFactors = F)

# To increase the output column in the test data

bank_test$y = NA

# To distinguish test and training data

bank_test$type = "test"
bank_train$type = "train"

#bank_train$train = NULL

bank_data <- rbind(bank_train,bank_test)

library(dplyr)

str(bank_data)

glimpse(bank_data)

# To find any missing values in complete data

apply(bank_data,2, function(x) sum(is.na(x)))

# No missing values found



CreateDummies=function(data,var,freq_cutoff=0){
  t=table(data[,var])
  t=t[t>freq_cutoff]
  t=sort(t)
  categories=names(t)[-1]
  
  for( cat in categories){
    name=paste(var,cat,sep="_")
    name=gsub(" ","",name)
    name=gsub("-","_",name)
    name=gsub("\\?","Q",name)
    name=gsub("<","LT_",name)
    name=gsub("\\+","",name)
    name=gsub("\\/","_",name)
    name=gsub(">","GT_",name)
    name=gsub("=","EQ_",name)
    name=gsub(",","",name)
    
    data[,name]=as.numeric(data[,var]==cat)
  }
  
  data[,var]=NULL
  return(data)
}


cat_col = c("job",'marital','education','default','housing','loan','contact','month','poutcome')

for(cat in cat_col){
  bank_data = CreateDummies(bank_data,cat,50)
}

glimpse(bank_data)
str(bank_data)

# To convert y column into factor as we want treat it as classification problem

bank_data$y= as.numeric(bank_data$y=="yes")

#bank_data$y=as.factor(bank_data$y) 
table(bank_data$y)

glimpse(bank_data)

bank_train=bank_data %>% filter(type=="train") %>% select(-type)
bank_test =bank_data %>% filter(type=="test") %>% select(-type,-y)  

# Creating two set of data in training data

set.seed(3)
s = sample(1:nrow(bank_train),0.8*nrow(bank_train))
bank_train1 = bank_train[s,]
bank_train2 = bank_train[-s,]


library(car)

for_vif =lm(y~.,data = bank_train1)
summary(for_vif)
sort(vif(for_vif),decreasing = T)

# Removing variable month_may

for_vif =lm(y~.-month_may,data = bank_train1)
summary(for_vif)
sort(vif(for_vif),decreasing = T)

# Removing variable job_blue_collar
for_vif =lm(y~.-month_may-job_blue_collar,data = bank_train1)
summary(for_vif)
sort(vif(for_vif),decreasing = T)

# Removing variable contact_unknown
for_vif =lm(y~.-month_may-job_blue_collar-contact_unknown,data = bank_train1)
summary(for_vif)
sort(vif(for_vif),decreasing = T)

#Removing VIF > 10

log_fit = glm(y~.-job_services-month_may-job_blue_collar-contact_unknown,data=bank_train1,family = "binomial")

log_fit = step(log_fit)

summary(log_fit)

library(pROC)

val.score = predict(log_fit,newdata = bank_train2,type = 'response')

auc_score = auc(roc(bank_train2$y,val.score))

auc_score

library(ggplot2)
mydata=data.frame(y=bank_train2$y,val.score=val.score)
ggplot(mydata,aes(y=y,x=val.score,color=factor(y)))+
  geom_point()+geom_jitter()

# Building model on entire dataset

for_vif =lm(y~.-month_may-job_blue_collar-contact_unknown,data = bank_train)
sort(vif(for_vif),decreasing = T)

log_fit_final = glm(y~.-month_may-job_blue_collar-contact_unknown,data=bank_train,family = "binomial")
summary(log_fit)

log_fit_final = step(log_fit_final)
formula(log_fit_final)

summary(log_fit_final)

train.score=predict(log_fit_final,newdata = bank_train,type = 'response')

real=bank_train$y
cutoffs=seq(0.001,0.999,0.001)
cutoff_data=data.frame(cutoff=99,Sn=99,Sp=99,KS=99,F5=99,F.1=99,M=99)

for(cutoff in cutoffs){
  
  predicted=as.numeric(train.score>cutoff)
  
  TP=sum(real==1 & predicted==1)
  TN=sum(real==0 & predicted==0)
  FP=sum(real==0 & predicted==1)
  FN=sum(real==1 & predicted==0)
  
  P=TP+FN
  N=TN+FP
  
  Sn=TP/P
  Sp=TN/N
  precision=TP/(TP+FP)
  recall=Sn
  
  KS=(TP/P)-(FP/N)
  F5=(26*precision*recall)/((25*precision)+recall)
  F.1=(1.01*precision*recall)/((.01*precision)+recall)
  
  M=(4*FP+FN)/(5*(P+N))
  
  cutoff_data=rbind(cutoff_data,c(cutoff,Sn,Sp,KS,F5,F.1,M))
}

cutoff_data=cutoff_data[-1,]

my_cutoff=cutoff_data$cutoff[which.max(cutoff_data$KS)]

my_cutoff

# To create confusion matrix to calculate KS score by using bank_train2 dataset

train.score1 = predict(log_fit_final,newdata = bank_train2,type = 'response')
train.predicted = as.numeric(train.score1>my_cutoff)
table(train.predicted)

library(caret)

confusionMatrix(as.factor(train.predicted),as.factor(bank_train2$y))


test.prob.score= predict(log_fit_final,newdata = bank_test,type='response')

test.predicted = as.numeric(test.prob.score>my_cutoff)

write.csv(test.predicted,"Dr_P_Adhikary_P5_part2.csv",row.names = F)

table(test.predicted)
