library(caret)
library(dplyr)
library(randomForest)
library(pROC)
library(ModelMetrics)



a=read.csv("ttrain.csv")
b=read.csv("test.csv")

a$marital=as.factor(a$marital)
a$job=as.factor(a$job)
a$education=as.factor(a$education)
a$default=as.factor(a$default)
a$housing=as.factor(a$housing)
a$loan=as.factor(a$loan)
a$contact=as.factor(a$contact)
a$poutcome=as.factor(a$poutcome)
a$balance=ifelse(a$balance<0,0,a$balance)
a$y=as.factor(a$y)



b$marital=as.factor(b$marital)
b$job=as.factor(b$job)
b$education=as.factor(b$education)
b$default=as.factor(b$default)
b$housing=as.factor(b$housing)
b$loan=as.factor(b$loan)
b$contact=as.factor(b$contact)
b$poutcome=as.factor(b$poutcome)
b$balance=ifelse(b$balance<0,0,a$balance)





train_up=upSample(a[,-17],a[,17],list=FALSE)
x_train=train_up[,c(1:16)]
y_train=train_up[,17]
y_train=as.data.frame(y_train)
colnames(y_train)="y"
a=cbind(x_train,y_train)



c=a
d=b

a=a[,-c(11,12)]
b=b[,-c(11,12)]


idx=createDataPartition(a$y,p=0.8,list=FALSE)
train=a[idx,-c(1,12)]
valid=a[-idx,-c(1,12)]
prepro=preProcess(train[,-13],method=c("range"))
scaled_train=predict(prepro,train)
scaled_valid=predict(prepro,valid)
scaled_test=predict(prepro,b)
scaled_ttrain=predict(prepro,a)



md_rf=randomForest(y~.,data=scaled_train,ntree=300,probability=TRUE,importance=T,na.action=na.omit)
ModelMetrics::rmse(md_rf)
rmse(md_rf)
roc(scaled_train$y,as.numeric(md_rf$predicted))
pred_md_rf=predict(md_rf,newdata=scaled_valid,probability=TRUE)
library(pROC)
roc(scaled_valid$y,as.numeric(pred_md_rf))
head(pred_md_rf)
valid_rf6=RMSE(scaled_valid$y,pred_md_rf)
ModelMetrics::rmse(scaled_valid$y,pred_md_rf)



rf=randomForest(y~.,data=scaled_ttrain,ntree=300,probability=TRUE)
roc(scaled_ttrain$y,as.numeric(rf$predicted))
pred_rf=predict(rf,newdata=scaled_test,probability=TRUE,type="prob")
head(pred_rf)
result=data.frame(ID=b$ID,predict=pred_rf[,2])



rf2=randomForest(y~.,data=train,ntree=300,probability=TRUE)
pred_rf2=predict(rf2,newdata=valid,probability=TRUE ,type="response")
roc(valid$y,as.numeric(pred_rf2))
head(pred_rf2)



rf1=randomForest(y~.,data=a,ntree=300,probability=TRUE)
roc(a$y,as.numeric(rf1$predicted))
pred_rf1=predict(rf1,newdata=b,probability=TRUE ,type="prob")
head(pred_rf1)



rf5=randomForest(y~.,data=c,ntree=300,probability=TRUE)
roc(c$y,as.numeric(rf5$predicted))
pred_rf5=predict(rf5,newdata=d,probability=TRUE ,type="prob")
head(pred_rf5)