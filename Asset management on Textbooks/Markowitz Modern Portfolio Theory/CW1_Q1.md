Q1：

​	![image-20220214124214491](C:\Users\Peanut Robot\AppData\Roaming\Typora\typora-user-images\image-20220214124214491.png)

#### Notation:



- Construct a risky loans portfolio $\Pi = \{w_i \text{ Loan}_i \}_{i=1}^3 $ with 

$$
\begin{aligned}
R_{\Pi} & = w^TR \\
\sigma^2_{\Pi} & = w^T \Sigma w \\

\end{aligned}
$$

where:
$$
\Sigma = \left[\begin{array}{cccc}
\sigma_1^2 & \rho_{12}\sigma_1\sigma_2 & \rho_{13}\sigma_1\sigma_3 \\
\rho_{21}\sigma_1\sigma_2 & \sigma_2^2 & \rho_{23}\sigma_2\sigma_3 \\
\rho_{31}\sigma_1\sigma_3 & \rho_{32}\sigma_2\sigma_3 & \sigma_3^2  \\

\end{array}\right]
$$

- Considering the risk-free asset,  construct a risky loans portfolio  $\Pi^*$  with
  $$
  \begin{aligned}
  R_{\Pi*} & = h^TR + (1 - h^T1_n)R_f = R_f +h^T(R-R_f1_n) \\
  \sigma^2_{\Pi*} & = h^T \Sigma h \\
  
  \end{aligned}
  $$

- Denote 
  $$
  1_n = \left[\begin{array}{cccc}
  1 \\ 1 \\...\\ 1
  
  \end{array}\right]
  $$

- $$
  \begin{aligned}
  
  \text{the market price of risk  } = &  \frac{  R_{\Pi*} - R_f}{ \sigma_{\Pi*} } \\
  \text{CML:   } R =  & \frac{  R_{\Pi*} - R_f}{ \sigma_{\Pi*} } \sigma + R_f
  \end{aligned}
  $$

  



#### Part 0: Optimization Problem ——find the maximum slope 



(1)  Solve the minimization problem:
$$
\begin{aligned}
&\max_{w}\frac{w^TR - R_f}{ (w^T\Sigma w)^{1/2}} \\

s.t.&\{
\begin{array}{c} 
w^T 1_n = 1 
\end{array}, \text{where }1_n \text{ is a identity matrix}
\\

\\
& L =  \frac{w^TR - R_f}{ (w^T\Sigma w)^{1/2}}  - \lambda(w^T 1_n - 1 )
\\

&
\left\{\begin{array}{l} 
\frac{\part L}{\part w} = 
	\frac{ R( w^T\Sigma w)^{1/2} -   (w^TR - R_f) \frac{1}{2}( w^T\Sigma w)^{-1/2}2\Sigma w }
	{ w^T\Sigma w} 
	- \lambda 1_n= 0......(1)   \\
\frac{\part L}{\part \lambda} =  w^T1_n -1 = 0 ...............(2)

\end{array}
\right. \\

\end{aligned}
$$
Solve $\lambda$
$$
\begin{aligned}
 R( w^T\Sigma w) -   (w^TR - R_f) \Sigma w  = & \lambda 1_n (w^T \Sigma  w)^{\frac{3}{2}} 
 \\
 
  (Rw^T\Sigma-w^TR\Sigma - R_f\Sigma)  w  = & \lambda 1_n (w^T \Sigma  w)^{\frac{3}{2}}

\\

 w^T (Rw^T\Sigma-w^TR\Sigma - R_f\Sigma)  w  = & \lambda  w^T1_n (w^T \Sigma  w)^{\frac{3}{2}}
 \\
  w^T (Rw^T\Sigma-w^TR\Sigma - R_f\Sigma)  w  = & \lambda  (w^T \Sigma  w)^{\frac{3}{2}}
  
\\

 \frac{ w^T (Rw^T\Sigma-w^TR\Sigma - R_f\Sigma)  w }{(w^T \Sigma  w)^{\frac{3}{2}}} = & \lambda

\end{aligned}
$$

Plug $\lambda$ back into (1)
$$
\begin{aligned}
 \frac{ R( w^T\Sigma w)^{1/2} -   (w^TR - R_f) \frac{1}{2}( w^T\Sigma w)^{-1/2}2\Sigma w }
	{ w^T\Sigma w} 
 =& \frac{ w^T (Rw^T\Sigma-w^TR\Sigma - R_f\Sigma)  w }{(w^T \Sigma  w)^{\frac{3}{2}}} 1_n

\\
R( w^T\Sigma w) -(w^TR - R_f) \Sigma w = & w^T (Rw^T\Sigma-w^TR\Sigma - R_f\Sigma)  w 1_n 

\\
(Rw^T\Sigma-w^TR\Sigma - R_f\Sigma)  w  = & w^T (Rw^T\Sigma-w^TR\Sigma - R_f\Sigma)  w 1_n 
\\


Aw =& \tr{(ww^TA)}1_n

\\
(A \times 1_nw^T) 1_n = & \tr{(ww^TA)}1_n \\


\end{aligned}
$$

(2) Now we find it could very hard to solve $w$ from the equation. So we think about alternative ways to get the close solution of $w$. 

- We know that the CML is the tangent of the efficient frontier, and CML passes through the risk-free point. Actually, the original optimization problem equals to find the tantency point. Thus, We can find two methods to replace the original optimization problem:

![image-20220220185920229](C:\Users\Peanut Robot\AppData\Roaming\Typora\typora-user-images\image-20220220185920229.png)

- **Method 1:**  

  (1) Firstly, we try to get the CML. In order to do that, we introduce a constant $R_{\Pi} $ to create an optimization problem:  
  $$
  \begin{aligned}
  &\min_{h} h^T\Sigma h \\
  
  s.t& \{ h^TR + (1 - h^T1_n)R_f = R_{\Pi*} 
  
  
  \end{aligned}
  $$
  Solve the optimization problem： we can get the CML: $ R_{\Pi*} = f(\text{Risk} )$ , and $h = h(R_{\Pi*})$

  

  (2) Secondly, we bring in some special conditions to solve the constant $R_{\Pi*} $ :

  At the tangency point, we know: 
  $$
  h^T 1_n =1
  $$
  Thus,
  $$
  h(R_{\Pi*}) 1_n =1
  $$
  And we can solve the constant $R_{\Pi*}$ , and then get $h$



- **Method 2:**

  (1) Firstly, we try to get the effective frontier. In order to do that, we introduce a constant $R_{\Pi} $ to create an optimization problem:  

$$
\begin{aligned}
 &\min_w w^T   \Sigma w \\

\\
s.t.&
\{
\begin{array}{c} 
w^T R = R_{\Pi} \\
w^T 1_n = 1 
\end{array}


\end{aligned}
$$

​		Solve the optimization problem： we can get the effective frontier: $\text{Risk}  = g( R_{\Pi} )$ , and $w = w(R_{\Pi,0})$

​		(2)  Secondly, we bring in some special conditions to solve the constant $R_{\Pi} $ :

​		Using the effective frontier, we can easily find the tangency point:  
$$
\max_{R_{\Pi}} \text{Slope} =  \frac{  R_{\Pi} - R_f}{\sigma_{\Pi}}  =  \frac{  R_{\Pi} - R_f}{g(R_{\Pi}) }
$$
​		And we can solve the constant $R_{\Pi} $, and then get $w$ 





We show the derivation process with more details in Part 1 & Part 2



#### Part 1:  derivation process about Method 1

(1) Firstly, we find the CML:
$$
\begin{aligned}
&\min_{h} h^T\Sigma h \\

s.t& \{ h^TR + (1 - h^T1_n)R_f = R_{\Pi*} 


\end{aligned}
$$
The constraints can be written as
$$
\begin{aligned}

h^TR + (1 - h^T1_n)R_f = & R_{\Pi*}  \\

R_f +h^T(R-R_f1_n) =& R_{\Pi*}

\\
h_T r= & r_0

\end{aligned}
$$
denote $r=R-R_f1_n$ and $r_0 = R_{\Pi*} -R_f$



- we solve the problem:

$$
\begin{aligned}
L &= h^T\Sigma h - \lambda(h^Tr -r_{0})\\

\\


&
\left\{\begin{array}{l} 
\frac{\part L}{\part h} = 
	2\Sigma h - 
	\lambda r= 0................(1)   \\
\frac{\part L}{\part \lambda} =  h^Tr -r_{0} = 0 ...............(2)

\end{array}
\right. \\

\end{aligned}
$$

​	Using the equation (1): 
$$
\begin{aligned}
2\Sigma h - \lambda r & = 0 \\

h & =\ \frac{\lambda}{2 } \Sigma^{-1}r

\end{aligned}
$$
Plug $h$ into the equation (2)
$$
\begin{aligned}


h & =\ \frac{\lambda}{2 } \Sigma^{-1}r \\

r_0= r^Th & = r^T  \frac{\lambda}{2 } \Sigma^{-1}r \\

\lambda & =  \frac{2 r_0}{ r^T \Sigma^{-1} r }


\end{aligned}
$$
​	Now plug $\lambda$ back into $ h  = \frac{\lambda}{2 } \Sigma^{-1}r $ :
$$
\begin{aligned}

h & =\ \frac{\lambda}{2 } \Sigma^{-1}r

\\
& = \frac{1}{2 } \Sigma^{-1}r \frac{2 r_0}{ r^T \Sigma^{-1} r }

\\
& = r_0 \frac{ \Sigma^{-1}r }{ r^T \Sigma^{-1} r }

\end{aligned}
$$


(2)  Secondly, we bring in some special conditions to solve the constant $r_0 $.  At the tangency point, we have the condition $1^T_nh = 1$ , we plug the condition into the  solution of $h$:

$$
\begin{aligned}

h & = r_0 \frac{ \Sigma^{-1}r }{ r^T \Sigma^{-1} r } \\

1=1^T_n h & = r_0 \frac{1^T_n \Sigma^{-1}r }{ r^T \Sigma^{-1} r } \\

r_0 &= \frac{ r^T \Sigma^{-1} r   }{ 1^T_n \Sigma^{-1}r}

\end{aligned}
$$

Now plug $ r_0 = \frac{ r^T \Sigma^{-1} r   }{ 1^T_n \Sigma^{-1}r} $ back into the solution of  $h$:
$$
\begin{aligned}

h & = r_0 \frac{ \Sigma^{-1}r }{ r^T \Sigma^{-1} r } \\

 &= \frac{ r^T \Sigma^{-1} r   }{ 1^T_n \Sigma^{-1}r}  \frac{ \Sigma^{-1}r }{ r^T \Sigma^{-1} r }
 
 
 \\
 & = \frac{ \Sigma^{-1} r   }{ 1^T_n \Sigma^{-1}r} 

\end{aligned}
$$
Now, we get the solution of risky portfolio (tangency point) weights: 
$$
\begin{aligned}

h & = \frac{ \Sigma^{-1} r   }{ 1^T_n \Sigma^{-1}r} 
 =\frac{ \Sigma^{-1} (R-R_f1_n)   }{ 1^T_n \Sigma^{-1}(R-R_f1_n)} 

\end{aligned}
$$


#### Part 2:  derivation process about Method 2

(1) Firstly, find the effective front:
$$
\begin{aligned}
 &\min_w w^T   \Sigma w \\

\\
s.t.&
\{
\begin{array}{c} 
w^T R = R_{\Pi} \\
w^T 1_n = 1 
\end{array}


\end{aligned}
$$

Solve the problem:
$$
\begin{aligned}
L & = w^T\Sigma w - \lambda_1(w^TR-R_{\Pi}) - \lambda_2(w^T1_n -1 ) \\

\\
&
\left\{\begin{array}{l} 
\frac{\part L}{\part w} = 2\Sigma w - \lambda_1 R - \lambda_2 1_n = 0......(1)   \\
\frac{\part L}{\part \lambda_1} = w^TR-R_{\Pi} =0    ............(2)\\ 
\frac{\part L}{\part \lambda_2} =  w^T1_n -1 = 0 ...............(3)

\end{array}
\right. \\

\\
\text{Now } & \text{substitute equation (1) into (2) (3) } \\
& w  = \frac{1}{2}\Sigma^{-1} (\lambda_1R+\lambda_2 1_n)\\

&
\left\{\begin{array}{l}
R_{\Pi}=\frac{1}{2}\left(\lambda_{1} R^{T} \Sigma^{-1} R+\lambda_{2} 1_{n}^{T} \Sigma^{-1} R\right) \\
1=\frac{1}{2}\left(\lambda_{1} R^{T} \Sigma^{-1} 1_{n}+\lambda_{2} 1_{n}^{T} \Sigma^{-1} 1_{n}\right)
\end{array}\right.
\\

&
\text{which can be rewrited as following: }
\\
&
\begin{aligned}

\left[\begin{array}{c}
R_{\Pi} \\
1
\end{array}\right]=\frac{1}{2} \cdot\left[\begin{array}{cc}
R^{T} \Sigma^{-1} R & 1_{n}^{T} \Sigma^{-1} R \\
R^{T} \Sigma^{-1} 1_{n} & 1_{n}^{T}{\Sigma}^{-1} 1_{n}
\end{array}\right]\left[\begin{array}{l}
\lambda_{1} \\
\lambda_{2}
\end{array}\right] ....(4)

\end{aligned}

\\
\\
\text{Now } & \text{substitute equation (1) into } w^T\Sigma w \\


&
\begin{aligned}
w^T\Sigma w 
&= \frac{1}{4}\left(\lambda_{1} R^{T} \Sigma^{-1}+\lambda_{2} 1_{n}^{T} \Sigma^{-1}\right) \cdot \Sigma \Sigma^{-1}\left(\lambda_{1} R+\lambda_{2} 1_{n}\right) \\
&=\frac{1}{4}\left(\lambda_{1}{ }^{2} R^{T} \Sigma^{-1} R+2 \lambda_{1} \lambda_{2} R^{T} \Sigma^{-1} 1_{n}+\lambda_{2}{ }^{2} 1_{n}^{T} \Sigma^{-1} 1_{n}\right) \\
&=\frac{1}{4}\left[\begin{array}{ll}
\lambda_{1} & \lambda_{2}
\end{array}\right]\left[\begin{array}{ll}
R^{T} \Sigma^{-1} R & R^{T} \Sigma^{-1} 1_{n} \\
R^{T} \Sigma^{-1} 1_{n} & 1_{n}^{T} \Sigma^{-1} 1_{n}
\end{array}\right]\left[\begin{array}{l}
\lambda_{1} \\
\lambda_{2}
\end{array}\right] .....(5) \\
\end{aligned}


\end{aligned}
$$
solve $\lambda_1 , \lambda_2$ from (4), and then substitute the solution into (5):
$$
\begin{aligned}
\sigma_{\Pi}^2 = & \left[\begin{array}{ll}
R_{\Pi} & 1
\end{array}\right]\left[\begin{array}{cc}
R^{T} \Sigma^{-1} R & 1_{n}^{T} \Sigma^{-1} R \\
R^{T} \Sigma^{-1} 1_{n} & 1_{n}^{T} \Sigma^{-1} 1_{n}
\end{array}\right]^{-1}\left[\begin{array}{cc}
R^{T} \Sigma^{-1} R & R^{T} \Sigma^{-1} 1_{n} \\
R^{T} \Sigma^{-1} 1_{n} & 1_{n}^{T} \Sigma^{-1} 1_{n}
\end{array}\right]


\left[\begin{array}{cc}
R^{T} \Sigma^{-1} R & 1_{n}^{T} \Sigma^{-1} R \\
R^{T} \Sigma^{-1} 1_{n} & 1_{n}^{T} \Sigma^{-1} 1_{n}
\end{array}\right]^{-1} 
\left[\begin{array}{c}
R_{\Pi} \\
1
\end{array}\right]
\\


= &  \left[\begin{array}{ll} R_{\Pi} & 1 \end{array}\right] 
\left[\begin{array}{cc}
R^{T} \Sigma^{-1} R & 1_{n}^{T} \Sigma^{-1} R \\
R^{T} \Sigma^{-1} 1_{n} & 1_{n}^{T} \Sigma^{-1} 1_{n}
\end{array}\right]^{-1} 
\left[\begin{array}{c} R_{\Pi} \\ 1 \end{array}\right] 

\\
= &\left[\begin{array}{ll} R_{\Pi} & 1 \end{array}\right] 
\left[\begin{array}{cc}
a & b  \\
b & c
\end{array}\right]^{-1} 
\left[\begin{array}{c} R_{\Pi} \\ 1 \end{array}\right] 

\\
= & \frac{cR_{\Pi}^2 - 2bR_{\Pi} +a}{ac-b^2}

\end{aligned}
$$

(2) Secondly, using the effective frontier, we can easily find the tangency point:  

$$
\begin{aligned}
   \max_{R_{\Pi}} \text{Slope} =&  \frac{  R_{\Pi} - R_f}
  
  { \sigma_{\Pi} }  \\
  
  \\
  \text{ solve the problem:} 
  
  \\
  
  \frac{\part\text{Slope}}{\part R_{\Pi}} =& \frac{
  \sigma_{\Pi} - \frac{\part \sigma_{\Pi}}{\part R_{\Pi}} (R_{\Pi} -R_f) 
  }{
  \sigma^2_{\Pi}
  }  =0
  \\
   \sigma_{\Pi} =&   \frac{\part \sigma_{\Pi}}{\part R_{\Pi}} (R_{\Pi} -R_f)  \\
   \\
   
   \frac{1}{R_{\Pi}-R_f} =
   =&  \frac{\part  \sigma_{\Pi}}{\sigma_{\Pi} \part R_{\Pi}} \\
  
  =&  \frac{\part  (\sigma_{\Pi}^2)^{1/2} }{\sigma_{\Pi} \part R_{\Pi}} \\
  
  =&  \frac{1}{\sigma_{\Pi}} \frac{1}{2} (\sigma_{\Pi}^2)^{-\frac{1}{2}}
  \frac{\part  \sigma_{\Pi}^2}{\part R_{\Pi}} \\
  
  = & (\frac{cR_{\Pi}^2 - 2bR_{\Pi} +a}{ac-b^2})^{-1} \frac{1}{2} \frac{2cR_{\Pi} - 2b}{ac-b^2}
  \\ 
  = &\frac{ac-b^2}{cR_{\Pi}^2 - 2bR_{\Pi} +a} \frac{cR_{\Pi} - b}{ac-b^2} \\
  = & \frac{cR_{\Pi} - b}{cR_{\Pi}^2 - 2bR_{\Pi} +a} \\
  \\
  
  cR_{\Pi}^2 - 2bR_{\Pi} +a = & cR_{\Pi}^2 - b R_{\Pi} - cR_f R_{\Pi} + b R_f \\
  
  (-b + cR_f)R_{\Pi} =& b R_f -a\\
  
  R_{\Pi} = & \frac{a-b R_f}{b - cR_f}
  
  
  \end{aligned}
$$

And then, we get the solution of risky portfolio (tangency point) weights: 
$$
\begin{aligned}
w 
=& \frac{1}{2} \Sigma^{-1}(\lambda_1R + \lambda_21_n) \\
=& \frac{1}{2} \Sigma^{-1} \

\left[\begin{array}{c}
 R  &
 1_n
\end{array}\right] 

\left[\begin{array}{l}
\lambda_{1} \\
\lambda_{2}
\end{array}\right] \\




= &\frac{1}{2} \Sigma^{-1} \

\left[\begin{array}{c}
 R  &
 1_n
\end{array}\right] 
2
\left[\begin{array}{cc}
R^{T} \Sigma^{-1} R & 1_{n}^{T} \Sigma^{-1} R \\
R^{T} \Sigma^{-1} 1_{n} & 1_{n}^{T}{\Sigma}^{-1} 1_{n}
\end{array}\right]^{-1}

\left[\begin{array}{c}
R_{\Pi} \\
1
\end{array}\right] \\
=
&  \Sigma^{-1} 
\left[\begin{array}{c}
 R  &
 1_n
\end{array}\right] 

\left[\begin{array}{c}
a & b \\ b & c
\end{array}\right]^{-1} 


\left[\begin{array}{c}
R_{\Pi} \\
1
\end{array}\right]

\\
= & 
\Sigma^{-1}

\left[\begin{array}{c}
R & 1_n
\end{array}\right]

(
\left[\begin{array}{c}
R^T \\ 1_n^T
\end{array}\right]

\Sigma^{-1}

\left[\begin{array}{c}
R & 1_n
\end{array}\right] 
)^{-1}

\left[\begin{array}{c}
R_{\Pi} \\ 1
\end{array}\right]

\\
\end{aligned}
$$


#### Part3: The procedure of Calculation:

we follow Method 1 to calculate the results:
$$
R =\frac{2}{100} \left[\begin{array}{c}
7 \\ 4 \\ 10
\end{array}\right]

\\
\Sigma^ =
\frac{9}{10000}
\left[\begin{array}{ccc}
4 & 1 & 2\\
1 & 1 &2 \\
2 & 2 & 25
\end{array}\right]
\\


\Sigma^{-1} =
\frac{10000}{9}
\left[\begin{array}{ccc}
\frac{1}{3} & -\frac{1}{3} & 0 \\
- \frac{1}{3} & \frac{32}{21} & -\frac{2}{21} \\
0 & -\frac{2}{21} & \frac{1}{21}
\end{array}\right]
$$

$$
\begin{aligned}
a &= R^T \Sigma ^{-1} R =  
\frac{2}{100} * \frac{2}{100} * \frac{10000}{9} \left[\begin{array}{c}
7 & 4 & 10
\end{array}\right]

\left[\begin{array}{ccc}
\frac{1}{3} & -\frac{1}{3} & 0 \\
- \frac{1}{3} & \frac{32}{21} & -\frac{2}{21} \\
0 & -\frac{2}{21} & \frac{1}{21}
\end{array}\right]

 \left[\begin{array}{c}
7 \\ 4 \\ 10
\end{array}\right]

= \frac{4}{9} 1^T_n \left[\begin{array}{ccc}
\frac{49}{3} & -\frac{28}{3} & 0 \\
- \frac{28}{3} & \frac{512}{21} & -\frac{80}{21} \\
0 & -\frac{80}{21} & \frac{100}{21}
\end{array}\right] 1_n = \frac{403}{21} \frac{4}{9}
\\


b &= R^T \Sigma ^{-1} 1_n
=
\frac{2}{100}  * \frac{10000}{9} \left[\begin{array}{c}
7 & 4 & 10
\end{array}\right]

\left[\begin{array}{ccc}
\frac{1}{3} & -\frac{1}{3} & 0 \\
- \frac{1}{3} & \frac{32}{21} & -\frac{2}{21} \\
0 & -\frac{2}{21} & \frac{1}{21}
\end{array}\right]

 \left[\begin{array}{c}
1 \\ 1 \\ 1
\end{array}\right]

=
\frac{200}{9} 1^T_n \left[\begin{array}{ccc}
\frac{7}{3} & -\frac{7}{3} & 0 \\
- \frac{4}{3} & \frac{128}{21} & -\frac{8}{21} \\
0 & -\frac{20}{21} & \frac{10}{21}
\end{array}\right] 1_n = \frac{82}{21} \frac{200}{9}

\\

c &= 1_n^T \Sigma ^{-1} 1_n

=
\frac{10000}{9} \left[\begin{array}{c}
1 & 1 & 1
\end{array}\right]

\left[\begin{array}{ccc}
\frac{1}{3} & -\frac{1}{3} & 0 \\
- \frac{1}{3} & \frac{32}{21} & -\frac{2}{21} \\
0 & -\frac{2}{21} & \frac{1}{21}
\end{array}\right]

 \left[\begin{array}{c}
1 \\ 1 \\ 1
\end{array}\right]

=
\frac{10000}{9} 1^T_n 
\left[\begin{array}{ccc}
\frac{1}{3} & -\frac{1}{3} & 0 \\
- \frac{1}{3} & \frac{32}{21} & -\frac{2}{21} \\
0 & -\frac{2}{21} & \frac{1}{21}
\end{array}\right]
1_n = \frac{22}{21} \frac{10000}{9}
\end{aligned}
$$

$$
\left[\begin{array}{ccc}
a & b \\
b & c
\end{array}\right] ^ {-1} 

=
(\frac{4}{9*21}
\left[\begin{array}{ccc}
403 & 82*50 \\
82*50 & 22*2500
\end{array}\right])^{-1} 

= 

\frac{9 *21}{4} *
\frac{1}{5355000}
\left[\begin{array}{ccc}
22*2500 & -82*50 \\
-82*50 & 403
\end{array}\right]
$$

$R_{\Pi}$: 
$$
R_{\Pi} =  \frac{a-b R_f}{b - cR_f} 
= \frac{403\times 4 - 82 \times 200 \times 0.05}{82 \times 200 - 22 \times 10000 \times0.05}
= \frac{403\times 4 - 82 \times  10}{82 \times 200 - 22 \times 500}
= \frac{1}{100}\frac{1612 -820}{164 -110} 
= \frac{792}{54} \%
= \frac{132}{900} 
$$


weights:
$$
\begin{aligned}
w = &  \Sigma^{-1} 
\left[\begin{array}{c}
 R  &
 1_n
\end{array}\right] 

\left[\begin{array}{c}
a & b \\ b & c
\end{array}\right]^{-1} 


\left[\begin{array}{c}
R_{\Pi} \\
1
\end{array}\right]


\\
= &
\frac{10000}{9}
\left[\begin{array}{ccc}
\frac{1}{3} & -\frac{1}{3} & 0 \\
- \frac{1}{3} & \frac{32}{21} & -\frac{2}{21} \\
0 & -\frac{2}{21} & \frac{1}{21}
\end{array}\right]


\frac{2}{100} \left[\begin{array}{c}
7 & 50\\ 
4 & 50\\ 
10 & 50
\end{array}\right]

\frac{9 *21}{4} *
\frac{1}{5355000}
\left[\begin{array}{ccc}
22*2500 & -82*50 \\
-82*50 & 403
\end{array}\right]

\frac{1}{900}
\left[\begin{array}{ccc}
132\\
900
\end{array}\right]


\\
=&
\frac{1}{18 \times 5355000} 

\left[\begin{array}{ccc}
7 & -7 & 0 \\
-7 &  32 & -2 \\
0 & -2 & 1
\end{array}\right]

\left[\begin{array}{c}
7 & 50\\ 
4 & 50\\ 
10 & 50
\end{array}\right]

\left[\begin{array}{ccc}
22*2500 & -82*50 \\
-82*50 & 403
\end{array}\right]

\left[\begin{array}{ccc}
132\\
900
\end{array}\right]

\\
= &
\frac{1 }{18 \times 5355000} 
\left[\begin{array}{ccc}
21 & 0 \\
59 & 1150 \\
2 & -50
\end{array}\right]

\left[\begin{array}{ccc}
35700\\
-1785
\end{array}\right]
100
\\

=&
\frac{1}{18 \times 53550} 
\left[\begin{array}{ccc}
749700\\
53550\\
160650
\end{array}\right]

\\
=&
\frac{1}{18} 
\left[\begin{array}{ccc}
14\\
1\\
3
\end{array}\right]

\end{aligned}
$$


$\sigma_{\Pi}^2$:
$$
\begin{aligned}
\sigma_{\Pi}^2 = & 

w^T\Sigma w 

\\
= &

\frac{1}{18} 
\left[\begin{array}{ccc}
14 & 1 & 3
\end{array}\right]

\frac{9}{10000}
\left[\begin{array}{ccc}
4 & 1 & 2\\
1 & 1 &2 \\
2 & 2 & 25
\end{array}\right]

\frac{1}{18} 
\left[\begin{array}{ccc}
14\\
1\\
3
\end{array}\right]

\\
= &
\frac{1}{36 \times 10000} 
1_n^T
\left[\begin{array}{ccc}
784 & 14 & 84\\
14 & 1 & 6 \\
84 & 6 & 225
\end{array}\right]
1_n 

\\
=& \frac{1218}{36 \times 10000} 

\\
= & \frac{203}{6} \%\%

\\
\\
\sigma_{\Pi} =& \sqrt{\frac{203}{6}} \% = 5.817 \%


\end{aligned}
$$
Slope:
$$
\begin{aligned}
\text{Slope} = & \frac{  R_{\Pi} - R_f}{\sigma_{\Pi}} 

\\

= & \frac{132/9 \% - 0.05}{ \sqrt{203/6}  \% }

\\

= & \frac{132/9 - 5}{ \sqrt{203/6}  }
\\
= & 1.662& 
\end{aligned}
$$
CML: 
$$
R = 1.662 \sigma + 0.05 
$$
The market price of risk = 1.662





![image-20220214125123295](C:\Users\Peanut Robot\AppData\Roaming\Typora\typora-user-images\image-20220214125123295.png)
$$
\begin{aligned}
0.02 ^2 = & t^2 \sigma^2_{\Pi}
\\
t^2 = &  4\%\%\frac{6}{203\%\%}  
\\
t = & \sqrt{\frac{24}{203}}
\\
= & 0.3438

\end{aligned}
$$


w:
$$
w_{new} =&
\left[\begin{array}{ccc}
\frac{14}{18}t\\
\frac{1}{18}t\\
\frac{3}{18}t \\
1 - t
\end{array}\right]
=

\left[\begin{array}{ccc}
0.2674\\
0.0191\\
0.0573 \\
0.6562
\end{array}\right]
$$
return
$$
R_{new} =

\left[\begin{array}{ccc}
\frac{14}{18}t &
\frac{1}{18}t &
\frac{3}{18}t &
\frac{18(1-t) }{18}
\end{array}\right]

\frac{1}{100}

\left[\begin{array}{ccc}
14\\
8\\
20 \\
5
\end{array}\right]

= \frac{1}{100} \frac{(196 + 8 + 60 -90)t +90}{18} = \frac{174 t + 90}{18} = 8.324 \%
$$