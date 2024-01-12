# For Lei, Chen, Lynch 2019 Consistent community detection in multi-layer network data.


GetCluster <- function(A, k, reps = 1, maxit = 50, verbose ='off'){
 # by  kehui chen @khchen@pitt.edu
  # A is p * m * m, this code only works for p > 1
  totsumDBest <- Inf
  idxBest <- NULL
  centerBest <- NULL
  dimvec <- dim(A)
  m <- dimvec[3]
  Avec <- t(matrix(A, dimvec[1]*dimvec[2], dimvec[3]))
  if (dim(unique(Avec))[1] <k) {
    return(list(idx = idxBest, center = centerBest))
  }
  for (rep in 1:reps){
    idx <- kmeans(Avec, k)$cluster 
    iter <- 0
    prevtotsumD <- Inf
    while (iter < maxit){
      if (verbose == 'on'){
         cat('step = ', iter, '\n' )
         }
      iter <- iter + 1
      # deal with cluster that have just lose all their members
      counts <- summary(factor(idx, 1:k))
      empties <- which(counts == 0)
      for (i in empties){
        from <- sample(which(counts > 1), 1,0)
        lonely <- sample(which(idx == from), 1, 0)
        idx[lonely] <- i
        counts[i] <- 1
        counts[from] <- counts[from]-1; 
      }   
      center <- GetCenter(A,idx,k)
      D <- GetDist(A, center[,idx,])
      totsumD <- sum(D[(idx-1)*m+(1:m)])
      if (prevtotsumD <  totsumD | prevtotsumD == totsumD){
        break    
      }
      previdx <- idx
      prevtotsumD <- totsumD
      prevcenter <- center
      nidx <- apply(D, 1, which.min)
      moved <- which(nidx != previdx)
      # resolve tie in favor of not move
      if (length(moved)>0) {
        moved <- moved[D[(previdx[moved]-1)*m+moved] > min(D[moved,])] 
      } 
      if (length(moved) ==0){
        break
      }
      idx[moved] = nidx[moved]
    }
    if (totsumD < totsumDBest){
      totsumDBest <- totsumD
      idxBest <-idx
      centerBest <- center
    }
  }
  return(idxBest)
  #return(list(idx = idxBest, center = centerBest))
}

GetCenter <-function(A, idx, k){
  p <- dim(A)[1]
  m.1 <- dim(A)[2]
   idx.1 <- idx[1:m.1]
   center <- array(0, dim=c(p,k,k))
   for (i in 1:k){
     for (j in i:k){
        if ((sum(idx.1 == i) ==1) & (sum(idx==j)==1) ) {
          center[,i,j] <-0.5
          center[,j,i] <- center[,i,j]
        } else{
        center[,i,j] <- apply(A[,idx.1==i,idx==j],1,mean)
        center[,j,i] <- center[,i,j]
        }
     }
   }
  return(center)
}

GetDist <- function(A, center){
  # A: p*m*m
  # center: p*m*k
  k <- dim(center)[3]
  m <- dim(A)[3]
  D <- matrix(0, m, k)
  for (j in 1:m){
      for (i in 1:k){
        D[j,i] <- norm( (A[,,j] - center[,,i]), 'F')^2
      }
  }
  return(D)
}

GetCluster.1 <- function(A, k, reps = 10, maxit = 100, verbose ='off'){
 # by  kehui chen @khchen@pitt.edu
  # A is m * m, this code only works for p = 1
  totsumDBest <- Inf
  idxBest <- NULL
  centerBest <- NULL
  dimvec <- dim(A)
  m <- dimvec[2]
  if (dim(unique(A))[1] <k) {
    return(list(idx = idxBest, center = centerBest))
  }
  for (rep in 1:reps){
    idx <- kmeans(A, k)$cluster 
    iter <- 0
    prevtotsumD <- Inf
    while (iter < maxit){
      if (verbose == 'on'){
         cat('step = ', iter, '\n' )
         }
      iter <- iter + 1
      # deal with cluster that have just lose all their members
      counts <- summary(factor(idx, 1:k))
      empties <- which(counts == 0)
      for (i in empties){
        from <- sample(which(counts > 1), 1,0)
        lonely <- sample(which(idx == from), 1, 0)
        idx[lonely] <- i
        counts[i] <- 1
        counts[from] <- counts[from]-1; 
      }   
      center <- GetCenter.1(A, idx,k)
      D <- GetDist.1(A, center[idx,])
      totsumD <- sum(D[(idx-1)*m+(1:m)])  
      if (prevtotsumD <  totsumD | prevtotsumD == totsumD){
        break    
      }
      previdx <- idx
      prevtotsumD <- totsumD
      prevcenter <- center
      nidx <- apply(D, 1, which.min)
      moved <- which(nidx != previdx)
      # resolve tie in favor of not move
      if (length(moved)>0) {
        moved <- moved[D[(previdx[moved]-1)*m+moved] > min(D[moved,])] 
      } 
      if (length(moved) ==0){
        break
      }
      idx[moved] = nidx[moved]
    }
    if (totsumD < totsumDBest){
      totsumDBest <- totsumD
      idxBest <-idx
      centerBest <- center
    }
  }
  return(list(idx = idxBest, center = centerBest))
}

GetCenter.1 <-function(A, idx, k){
   m.1 <- dim(A)[2]
   idx.1 <- idx[1:m.1]
   center <- matrix(0, k, k)
   for (i in 1:k){
     for (j in i:k){
        if ((sum(idx == i) ==1) & (sum(idx.1==j)==1) ) {
          center[i,j] <- 0.5
          center[j,i] <- center[i,j]
        } else if (sum(idx.1 == j) == 0 ){
 	     center[i,j] <- mean(A[idx==i,])
 		 center[j,i] <- center[i,j]
 	   } else
 	   {
        center[i,j] <- mean(A[idx==i,idx.1==j])
        center[j,i] <- center[i,j]
        }
     }
   }
   return(center)
 }


GetDist.1 <- function(A, center){
  # A: m*m.1
  # center: m.1*k  
  m <- dim(A)[1]
  k <- dim(center)[2]
  if (is.null(k)) {
    k <- 1
	D <- matrix(0, m, k)
	for (j in 1:m){
       D[j] <- sum((t(A[j,]) - center)^2)
    }	
  } else { 
    D <- matrix(0, m, k)
    for (j in 1:m){
       for (i in 1:k){
        D[j,i] <- sum((t(A[j,]) - center[,i])^2)
       }
    }
  }
  return(D)
}

Generate.theta <- function(n, K, clust.size){
  theta <- matrix(0, n, K)
  for (k in 1:K){
      if (k==1){id1 <-1} else {id1 <- sum(clust.size[1:(k-1)])+1}
      id2 <- sum(clust.size[1:k])
      theta[id1:id2, k] <- 1
  } 
  return(theta)
}

Generate.data <- function(Theta, Btensor, self = 0) {
  # generates random adjacency matrix
  p <- dim(Btensor)[1]
  n <- dim(Theta)[1]
  A <- array(0, dim = c(p, n,n))
  for (i in 1:p){ 
     P <- Theta%*%Btensor[i,,]%*%t(Theta)
     lower.tri.ind <- lower.tri(P)
     p.upper <- P[!lower.tri.ind]
     A.upper <- rbinom(n*(n+1)/2, 1, p.upper)
     A0 <- matrix(0, ncol = n, nrow = n)
     A0[!lower.tri.ind] <- A.upper
	 tA0 = t(A0)
     A0[lower.tri.ind]  <- tA0[lower.tri.ind]
	 if (self == 0){
	   diag(A0) <- 0
	 }
	 A[i,,] = A0
  }
  return(A)
}


Acomb <- function(A, d = 1){
     p = dim(A)[1]
	 n = dim(A)[2]
     B <- matrix(A, p, n*n)
	 U = svd(B)$u
	 U = t(U[,1:d])
	 B2 = U%*%B;
	 B2 = apply(B2, 2, mean)
	 return(matrix(B2, n, n)) 
}

Acbind <- function(A){
     B <- A[1,,]
	 p <- dim(A)[1]
	 for (i in 2:p){
	 B <- cbind(B, A[i,,]) 
   }
   return(B)
}
Pcbind <- function(P1, p){
	 B <- P1
	 for (i in 2:p){
	 B <- cbind(B, P1) 
   }
   return(B)
}

Pcbind.2 <- function(A, Theta){
    B <- Theta%*%A[1,,]%*%t(Theta)
	p <- dim(A)[1]
	 for (i in 2:p){
	 B <- cbind(B, Theta%*%A[i,,]%*%t(Theta)) 
   }
   return(B)
}
