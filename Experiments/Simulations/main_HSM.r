####################################################################################

# First use trace to change the source code so that we can get obj val for each iteration.
library(hsm)
library(igraph)

trace(hsm edit = TRUE)



#####################################################################################

function (y, lam, w = NULL, map, var, assign = NULL, w.assign = NULL, 
          get.penalval = FALSE, tol = 1e-08, maxiter = 10000, beta.ma = NULL) 
{
  if (is.null(assign)) {
    stopifnot(is.matrix(map))
    stopifnot(is.list(var))
    stopifnot(length(y) == max(c(var, recursive = TRUE), 
                               na.rm = TRUE))
    paths.result <- paths(map = map, var = var, w = w)
    assign <- paths.result$assign
    w.assign <- paths.result$w.assign
  }
  stopifnot(is.matrix(assign))
  stopifnot(is.list(w.assign))
  stopifnot(length(w.assign) == nrow(assign))
  stopifnot(length(y) == ncol(assign))
  p <- length(y)
  n.paths <- nrow(assign)
  beta0 <- beta1 <- NULL
  obj_val <- replicate(maxiter, 0)
  time <- replicate(maxiter, 0)
  if (is.null(beta.ma)) {
    beta0 <- beta1 <- matrix(0, ncol = p, nrow = n.paths)
  }
  else {
    beta0 <- beta1 <- beta.ma
  }
  penalval <- rep(0, n.paths)
  continue <- TRUE
  ite <- 0
  while (continue) {
    start_time <- Sys.time()
    if (get.penalval) {
      for (i in 1:n.paths) {
        out <- .C("pathgraph_prox2", r = as.double(y - 
                                                     colSums(beta1[-i, , drop = FALSE])), as.double(lam), 
                  as.double(w.assign[[i]]), as.integer(assign[i, 
                                                              ]), as.integer(p), as.integer(max(assign[i, 
                                                                                                       ])), pen = as.double(penalval[i]))
        beta1[i, ] <- out$r
        penalval[i] <- out$pen
      }
    }
    else {
      for (i in 1:n.paths) {
        out <- .C("pathgraph_prox", r = as.double(y - 
                                                    colSums(beta1[-i, , drop = FALSE])), as.double(lam), 
                  as.double(w.assign[[i]]), as.integer(assign[i, 
                                                              ]), as.integer(p), as.integer(max(assign[i, 
                                                                                                       ])))
        beta1[i, ] <- out$r
      }
    }
    end_time <- Sys.time()
    if (max(abs(beta0 - beta1)) < tol || ite > maxiter) 
      continue <- FALSE
    beta0 <- beta1
    ite <- ite + 1
    obj_val[ite] <- 0.5 * norm(colSums(beta1) - y, type = c("2"))^2 + 
      sum(penalval)
    time[ite] <- end_time - start_time
  }
  return(list(beta = colSums(beta1), ite = ite, penalval = ifelse(get.penalval, 
                                                                  sum(penalval), NA), assign = assign, w.assign = w.assign, 
              beta.ma = beta1, obj_val = obj_val, time = time))
}


#####################################################################################
# Two layer tree

maxiter<-3000
map <- cbind(replicate(100,1), 2:101)
var <- as.list(1:101)
paths(map, var)

Y <- read.csv(file = 'Data/y_two_tree.csv', header = FALSE)
Y <- as.matrix(Y)
beta_true <-read.csv(file = 'Data/beta_two_tree.csv',header = FALSE)
beta_true <-as.matrix(beta_true)
beta_res <-matrix(0,10,maxiter)
obj_val <-matrix(0,10,maxiter)

iter <- replicate(10,0)
time <- matrix(0,10,maxiter)
pena <- replicate(10,0)
obj <- replicate(10,0)
for (i in 1:10) {
  y <- Y[,i]
  
  result <- hsm(y=y, lam=0.1, map=map, var=var, get.penalval=TRUE, tol=1e-10, maxiter=maxiter)
  obj_val[i,] <-result$obj_val[1:maxiter]
  time[i,] <-result$time[1:maxiter]
}

write.csv(obj_val,file = "Result_R/objval_two_tree.csv")
write.csv(time,file = "Result_R/time_two_tree.csv")


#####################################################################################
# Two path
map1 <- matrix(c(1,2,1,52),ncol = 2, byrow = TRUE)
map2 <- cbind(2:50, 3:51)
map3 <- cbind(52:100, 53:101)
map <- do.call(rbind,list(map1,map2,map3))

var <- as.list(1:101)
paths(map, var)

Y <- read.csv(file = 'Data/y_two_path_tree.csv', header = FALSE)
Y <- as.matrix(Y)
beta_true <-read.csv(file = 'Data/beta_binary_tree.csv',header = FALSE)
beta_true <-read.csv(file = 'Data/beta_two_tree.csv',header = FALSE)
beta_true <-as.matrix(beta_true)
beta_res <-matrix(0,10,maxiter)
obj_val <-matrix(0,10,maxiter)

iter <- replicate(10,0)
time <- matrix(0,10,maxiter)
pena <- replicate(10,0)
obj <- replicate(10,0)
for (i in 1:10) {
  y <- Y[,i]
  
  result <- hsm(y=y, lam=0.1, map=map, var=var, get.penalval=TRUE, tol=1e-10, maxiter=maxiter)
  obj_val[i,] <-result$obj_val[1:maxiter]
  time[i,] <-result$time[1:maxiter]
}
write.csv(obj_val,file = "Result_R/objval_two_path_tree.csv")
write.csv(time,file = "Result_R/time_two_path_tree.csv")

#####################################################################################

# Binary tree
map <- cbind(2:127 %/%2, 2:127)

var <- as.list(1:127)
paths(map, var)

Y <- read.csv(file = 'Data/y_binary_tree.csv', header = FALSE)
Y <- as.matrix(Y)
beta_true <-read.csv(file = 'Data/beta_binary_tree.csv',header = FALSE)
beta_true <-read.csv(file = 'Data/beta_two_tree.csv',header = FALSE)
beta_true <-as.matrix(beta_true)
beta_res <-matrix(0,10,maxiter)
obj_val <-matrix(0,10,maxiter)

iter <- replicate(10,0)
time <- matrix(0,10,maxiter)
pena <- replicate(10,0)
obj <- replicate(10,0)
for (i in 1:10) {
  y <- Y[,i]
  
  result <- hsm(y=y, lam=0.1, map=map, var=var, get.penalval=TRUE, tol=1e-10, maxiter=maxiter)
  obj_val[i,] <-result$obj_val[1:maxiter]
  time[i,] <-result$time[1:maxiter]
}

write.csv(obj_val,file = "Result_R/objval_binary_tree.csv")
write.csv(time,file = "Result_R/time_binary_tree.csv")

#####################################################################################
# Reverse binary tree
map <-read.csv(file='Data/map_reverse_binary_127.csv',header = FALSE)
map <- as.matrix(map)
var <- as.list(1:127)
paths(map, var)
edge=as.vector(t(map))
g1 <-graph(edge)

plot(g1, edge.arrow.size=0.2,vertex.color = 'white',vertex.size=12, edge.color = 'grey', vertex.label.color = 'black')
Y <- read.csv(file = 'Data/y_reverse_binary_127.csv', header = FALSE)

obj_val <-matrix(0,10,maxiter)

iter <- replicate(10,0)
time <- matrix(0,10,maxiter)
pena <- replicate(10,0)
obj <- replicate(10,0)
for (i in 1:10) {
  y <- Y[,i]
  
  result <- hsm(y=y, lam=0.1, map=map, var=var, get.penalval=TRUE, tol=1e-10, maxiter=maxiter)
  obj_val[i,] <-result$obj_val[1:maxiter]
  time[i,] <-result$time[1:maxiter]
}
write.csv(obj_val,file = "Result_R/objval_reverse_binary_127.csv")
write.csv(time,file = "Result_R/time_reverse_binary_127.csv")

#####################################################################################
# Random DAG n100, r100
map <-read.csv(file='Data/map_DAG_100.csv',header = FALSE)
map <- as.matrix(map)
var <- as.list(1:100)
paths(map, var)
edge=as.vector(t(map))
g1 <-graph(edge)

l <- layout_with_fr(g1)

plot(g1, edge.arrow.size=0.2,vertex.color = 'white',vertex.size=5, edge.color = 'grey', layout=l,vertex.label.color = 'black',vertex.label=NA)



Y <- read.csv(file = 'Data/y_DAG_100.csv', header = FALSE)

obj_val <-matrix(0,10,maxiter)

iter <- replicate(10,0)
time <- matrix(0,10,maxiter)
pena <- replicate(10,0)
obj <- replicate(10,0)
for (i in 1:10) {
  y <- Y[,i]
  
  result <- hsm(y=y, lam=0.1, map=map, var=var, get.penalval=TRUE, tol=1e-10, maxiter=maxiter)
  obj_val[i,] <-result$obj_val[1:maxiter]
  time[i,] <-result$time[1:maxiter]
}
write.csv(obj_val,file = "Result_R/objval_DAG_100.csv")
write.csv(time,file = "Result_R/time_DAG_100.csv")