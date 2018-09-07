setwd("/Users/stephane/Desktop/Canu/CAP_2018/competition/data_as_download")
setwd("/Users/stephane/Desktop/Canu/CAP_2018/competition/data")

D = read.csv(file="train_cap2018.csv", header=TRUE, sep=",")
D = read.csv(file="test_cap2018.csv", header=TRUE, sep=",")

summary(D)
str(D)

yl = D$level1

n = 27310
X = as.matrix(D[,c(2:51,54:59)])  # how to deal with NA

#na.omit(D);
#X = as.matrix(D);

y = yl

train <- sample(nrow(X),nrow(X)*2/3)
xtrain <- X[ train,] ; ytrain <- y[ train]
xtest  <- X[-train,] ; ytest  <- y[-train]


library("e1071")
?svm
C_svm = c(A1=1, A2=2, B1=4, B2=10, C1=20, C2=44)
svm_model <- tune.svm(x=xtrain,y=ytrain,type="C-classification", gamma = c(0.01,0.025,0.05), cost = c(1, 10, 100), class.weights = C_svm, cachesize=2000, tolerance = 0.01, scale=TRUE)$best.model
svm_model_0 <- e1071::svm(x=xtrain,y=ytrain,type="C-classification", gamma = 0.025, cost = 10, class.weights = C_svm, cachesize=2000, tolerance = 0.01, scale=TRUE)

yp <- predict(svm_model,xtest)

nt = length(yp)
conft <- table(yp, ytest[1:nt])
conft
confM = as.matrix(conft)
nberr <- sum(conft - diag(diag(conft)))
perf <- nberr/nt
perf

costM = (matrix(c( 0, 1, 2, 3, 4, 6, 
                  1, 0, 1, 4, 5, 8,  
                  3, 2, 0, 3, 5, 8,  
                 10, 7, 5, 0, 2, 7, 
                 20, 16, 12, 4, 0, 8,  
                 44, 38, 32, 19, 13, 0),nrow=6,ncol=6)) 

cost = 100*sum(costM*confM)/sum(confM)
cost


Eval_CAp_2018 <- function(y_estimated,y_real){
  # evaluationfuiction for CAp competition
  conft <- table(yp, ytest[1:nt])
  confM = as.matrix(conft)
  costM = (matrix(c( 0, 1, 2, 3, 4, 6, 
                     1, 0, 1, 4, 5, 8,  
                     3, 2, 0, 3, 5, 8,  
                     10, 7, 5, 0, 2, 7, 
                     20, 16, 12, 4, 0, 8,  
                     44, 38, 32, 19, 13, 0),nrow=6,ncol=6)) 
  C = 100*sum(costM*confM)/sum(confM)  
  return(C)
}

cost2 = Eval_CAp_2018(yp, ytest[1:nt])


