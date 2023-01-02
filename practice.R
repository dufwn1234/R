##파일 불러오기
a=read.csv("Loan payments data.csv",fileEncoding="UTF-8-BOM")
##열 삭제
a=a[,-c(7,8)]

##데이터 확인
summary(a)

##데이터 전처리
a$loan_status=ifelse(a$loan_status=="PAIDOFF","PAIDOFF","no")
a$loan_status=as.factor(a$loan_status)
a$education=as.factor(a$education)
a$Gender=as.factor(a$Gender)
a$effective_date=as.factor(a$effective_date)
a$due_date=as.factor(a$due_date)

##데이터 분할
set.seed(2108)
library(caret)
idx=sample(1:nrow(a),size=nrow(a)*0.7)
train=a[idx,-1]
test=a[-idx,-c(1,2)]##아이디와 loan_status제외
test1=a[-idx,-2]    ##loan_status제외

##다른방법으로 분할
idx2=createDataPartition(train$loan_status,p=0.8,list=FALSE)
ttrain=train[idx2,]
valid=train[-idx2,]

##데이터 정규화
prepro=preProcess(ttrain[,-1],method="range")
scaled_ttrain=predict(prepro,newdata=ttrain)
scaled_valid=predict(prepro,newdata=valid)
##데이터 표준화
prepro1=preProcess(ttrain[,-1],method=c("center","scale"))
scaled_ttrain1=predict(prepro1,newdata=ttrain)
scaled_valid1=predict(prepro1,newdata=valid)

##SVM
set.seed(2108)
library(e1071)
md_svm=svm(loan_status~.,data=scaled_ttrain,probability=TRUE)  ## 정규화한 데이터
pred_md_svm=predict(object =md_svm,newdata=scaled_valid,probability=TRUE)
library(pROC)
roc(scaled_valid$loan_status,as.numeric(pred_md_svm))        ##ROC 
caret::confusionMatrix(pred_md_svm,scaled_valid$loan_status) ##혼동행렬

md_svm1=svm(loan_status~.,data=ttrain,probability=TRUE)      ##표준화, 정규화 안한 데이터
pred_md_svm1=predict(md_svm1,newdata=valid,probability=TRUE)
roc(valid$loan_status,as.numeric(pred_md_svm1))
caret::confusionMatrix(pred_md_svm1,valid$loan_status)  ##혼동행렬

md_svm2=svm(loan_status~.,data=scaled_ttrain1,probability=TRUE)    ## 표준화한 데이터
pred_md_svm2=predict(md_svm2,newdata=scaled_valid1,probability=TRUE)
roc(scaled_valid1$loan_status,as.numeric(pred_md_svm2))
caret::confusionMatrix(pred_md_svm2,scaled_valid1$loan_status)  ##혼동행렬


##randomForest
set.seed(2108)
library(randomForest)
md_rf=randomForest(loan_status~.,data=scaled_ttrain,ntree=300,probability=TRUE) ## 정규화한 데이터
roc(scaled_ttrain$loan_status,as.numeric(md_rf$predicted))
pred_md_rf=predict(md_rf,newdata=scaled_valid,probability=TRUE,type="response")
roc(scaled_valid$loan_status,as.numeric(pred_md_rf))
caret::confusionMatrix(pred_md_rf,scaled_valid$loan_status) ##혼동행렬

md_rf1=randomForest(loan_status~.,data=ttrain,ntree=300,probability=TRUE)  ##표준화, 정규화 안한 데이터
pred_md_rf1=predict(md_rf1,newdata=valid,probability=TRUE,type="response")
roc(valid$loan_status,as.numeric(pred_md_rf1))  ##ROC 

md_rf2=randomForest(loan_status~.,data=scaled_ttrain1,ntree=300,probability=TRUE)  ## 표준화한 데이터
pred_md_rf2=predict(md_rf2,newdata=scaled_valid1,probability=TRUE,type="response")
roc(scaled_valid1$loan_status,as.numeric(pred_md_rf2))  ##ROC 


scaled_train=predict(prepro,newdata=train)
##테스트 데이터 스케일링
scaled_test=predict(prepro,newdata=test)

##SVM으로 결과표현
re_svm=svm(loan_status~.,data=scaled_train,probability=TRUE)
pred_svm=predict(object =re_svm,newdata=scaled_test,probability=TRUE,type="prob")
print(pred_svm)

##원본 데이터로 해보기
re_svm2=svm(loan_status~.,data=train,probability=TRUE)
pred_svm2=predict(re_svm2,newdata=test,probability=TRUE,type="prob")
print(pred_svm2)