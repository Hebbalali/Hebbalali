---
layout: post
title:  "Bayesian optimization using Deep Gaussian processes"
date:   2023-03-21 21:29:06 +0100
categories: jekyll update
---


# Bayesian optimization using Deep Gaussian processes


```python
import numpy as np
import tensorflow_probability as tfp
import tensorflow as tf
import math
import time
import pyDOE
import scipy
from dgp_dace.Infill_criteria import *
import gpflow
from gpflow.base import Module, Parameter
from dgp_dace.models.dgp import DGP
from dgp_dace.BO.SO_BO import SO_BO
import matplotlib.pyplot as plt
```

First the problem has to be defined. For that the problems are defined as a python class.
This class has to specify if the problem is constrained, its input dimension and the evaluation. 

## Unconstrained case


```python
def f_step(x):
    return np.where(x>0.25,1.,0.)
```


```python
class Constrained_problem(object):
    def __init__(self):
        self.constraint = True
        self.dim = 1
    def fun(self,x):
        return [(x-0.5)**2,f_step(x)]
    
```


```python
problem =  Constrained_problem()
```


```python
Ns = 300
Xs = np.linspace(0., 1., Ns)[:, None]
Ys = problem.fun(Xs)[0]
Cs =  problem.fun(Xs)[1]
```


```python
plt.figure()
plt.plot(Xs,Ys,label='Objective function')
plt.plot(Xs,Cs,label='Constraint function')
plt.legend()
```




    <matplotlib.legend.Legend at 0x7f3f804153c8>




![png](https://hebbalali.github.io/Hebbalali/assets/output_8_1.png)


SO_BO is a Module class for single objective Unconstrained/constrained Bayesian Optimization using Gaussian processes and deep Gaussian 
processes. 
The following optimization problem is considered:
$$        
\min \mbox{     }    f(x)\\
\text{s.t.} \mbox{     }   x \in [0,1]^d\\
 \mbox{               }        g_i(x) \leq 0 \mbox{     } \forall i \in1,\ldots,n_c\\
    $$


A dictionary which defines the architecture of the objective function model.

It has the following form:


        {'layers':l, 'num_units':[q_1,q_2,\ldots,q_l],kernels:['rbf','matern32','matern52',...],'num_samples':S}
 


```python
model_C_dic = {'num_layers':2,'num_units':1,'kernels':'rbf','num_samples':10}
```

In this case the different layers will have the same kernel and the same number of units. If we want the layers to be defined differently the kernels have to be given as a list of string and num_units as a list of integer:


```python
model_C_dic = {'num_layers':2,'num_units':[2,1],'kernels':['rbf','matern32','matern52'],'num_samples':10}
```


```python
model_C_dic = {'num_layers':2,'num_units':1,'kernels':'rbf','num_samples':10}
```


```python
model_Y_dic = {'num_layers':0,'kernels':'rbf'}
```

In the case of available X,Y it can be given as arguments for the SO_BO class. Otherwise, the DoE_size is given and the X,Y are generated in the initialization of the SO_BO class.


```python
SO_BO?
```


```python
BO = SO_BO(problem,DoE_size=5,model_Y_dic=model_Y_dic,model_C_dic=model_C_dic,normalize_input=True)

```

    The DGP architecture
    layer 1 : dim_in 1 --> dim_out 1
    layer 2 : dim_in 1 --> dim_out 1
    


```python

```


```python
from gpflow.utilities import print_summary
print_summary(BO,fmt='notebook')
```


<table>
<thead>
<tr><th>name                                           </th><th>class    </th><th>transform       </th><th>prior  </th><th>trainable  </th><th>shape    </th><th>dtype  </th><th>value                  </th></tr>
</thead>
<tbody>
<tr><td>SO_BO.model_Y.kernel.variance                  </td><td>Parameter</td><td>Softplus        </td><td>       </td><td>True       </td><td>()       </td><td>float64</td><td>1.0                    </td></tr>
<tr><td>SO_BO.model_Y.kernel.lengthscales              </td><td>Parameter</td><td>Softplus        </td><td>       </td><td>True       </td><td>(1,)     </td><td>float64</td><td>[1.]                   </td></tr>
<tr><td>SO_BO.model_Y.likelihood.variance              </td><td>Parameter</td><td>Softplus + Shift</td><td>       </td><td>True       </td><td>()       </td><td>float64</td><td>1.0000000000000004e-05 </td></tr>
<tr><td>SO_BO.model_C[0].likelihood.likelihood.variance</td><td>Parameter</td><td>Softplus + Shift</td><td>       </td><td>True       </td><td>()       </td><td>float64</td><td>1.0                    </td></tr>
<tr><td>SO_BO.model_C[0].layers[0].q_mu                </td><td>Parameter</td><td>Identity        </td><td>       </td><td>True       </td><td>(5, 1)   </td><td>float64</td><td>[[0....                </td></tr>
<tr><td>SO_BO.model_C[0].layers[0].q_sqrt              </td><td>Parameter</td><td>FillTriangular  </td><td>       </td><td>True       </td><td>(1, 5, 5)</td><td>float64</td><td>[[[1.0000005, 0., 0....</td></tr>
<tr><td>SO_BO.model_C[0].layers[0].feature.Z           </td><td>Parameter</td><td>Identity        </td><td>       </td><td>True       </td><td>(5, 1)   </td><td>float64</td><td>[[-0.24493691...       </td></tr>
<tr><td>SO_BO.model_C[0].layers[0].kern.variance       </td><td>Parameter</td><td>Softplus        </td><td>       </td><td>True       </td><td>()       </td><td>float64</td><td>1.0                    </td></tr>
<tr><td>SO_BO.model_C[0].layers[0].kern.lengthscales   </td><td>Parameter</td><td>Softplus        </td><td>       </td><td>True       </td><td>(1,)     </td><td>float64</td><td>[1.]                   </td></tr>
<tr><td>SO_BO.model_C[0].layers[1].q_mu                </td><td>Parameter</td><td>Identity        </td><td>       </td><td>True       </td><td>(5, 1)   </td><td>float64</td><td>[[0....                </td></tr>
<tr><td>SO_BO.model_C[0].layers[1].q_sqrt              </td><td>Parameter</td><td>FillTriangular  </td><td>       </td><td>True       </td><td>(1, 5, 5)</td><td>float64</td><td>[[[1.0000005, 0., 0....</td></tr>
<tr><td>SO_BO.model_C[0].layers[1].feature.Z           </td><td>Parameter</td><td>Identity        </td><td>       </td><td>True       </td><td>(5, 1)   </td><td>float64</td><td>[[-0.24493691...       </td></tr>
<tr><td>SO_BO.model_C[0].layers[1].kern.variance       </td><td>Parameter</td><td>Softplus        </td><td>       </td><td>True       </td><td>()       </td><td>float64</td><td>1.0                    </td></tr>
<tr><td>SO_BO.model_C[0].layers[1].kern.lengthscales   </td><td>Parameter</td><td>Softplus        </td><td>       </td><td>True       </td><td>(1,)     </td><td>float64</td><td>[1.]                   </td></tr>
<tr><td>SO_BO.model_C[0].layers[2].q_mu                </td><td>Parameter</td><td>Identity        </td><td>       </td><td>True       </td><td>(5, 1)   </td><td>float64</td><td>[[0....                </td></tr>
<tr><td>SO_BO.model_C[0].layers[2].q_sqrt              </td><td>Parameter</td><td>FillTriangular  </td><td>       </td><td>True       </td><td>(1, 5, 5)</td><td>float64</td><td>[[[1.0000005, 0., 0....</td></tr>
<tr><td>SO_BO.model_C[0].layers[2].feature.Z           </td><td>Parameter</td><td>Identity        </td><td>       </td><td>True       </td><td>(5, 1)   </td><td>float64</td><td>[[-0.24493691...       </td></tr>
<tr><td>SO_BO.model_C[0].layers[2].kern.variance       </td><td>Parameter</td><td>Softplus        </td><td>       </td><td>True       </td><td>()       </td><td>float64</td><td>1.0                    </td></tr>
<tr><td>SO_BO.model_C[0].layers[2].kern.lengthscales   </td><td>Parameter</td><td>Softplus        </td><td>       </td><td>True       </td><td>(1,)     </td><td>float64</td><td>[1.]                   </td></tr>
</tbody>
</table>


### Feasible solutions


```python
BO.Xfeasible
```




    array([0.0834044])




```python
BO.Yfeasible
```




    array([0.17355189])



### Handling of the models within the SO_BO class

To use an existing model for the objective function


```python
kernel = gpflow.kernels.SquaredExponential(lengthscales=[1.0]*BO.X_train.shape[1],variance=1.0) 
model = gpflow.models.GPR((BO.X_train, BO.Y_train), kernel, noise_variance=1e-5)

```


```python
BO.model_Y = model
```


```python
BO.model_Y
```




&lt;gpflow.models.gpr.GPR object at 0x7fd18c3d9ef0&gt;
<table>
<thead>
<tr><th>name                   </th><th>class    </th><th>transform       </th><th>prior  </th><th>trainable  </th><th>shape  </th><th>dtype  </th><th>value                 </th></tr>
</thead>
<tbody>
<tr><td>GPR.kernel.variance    </td><td>Parameter</td><td>Softplus        </td><td>       </td><td>True       </td><td>()     </td><td>float64</td><td>1.0                   </td></tr>
<tr><td>GPR.kernel.lengthscales</td><td>Parameter</td><td>Softplus        </td><td>       </td><td>True       </td><td>(1,)   </td><td>float64</td><td>[1.]                  </td></tr>
<tr><td>GPR.likelihood.variance</td><td>Parameter</td><td>Softplus + Shift</td><td>       </td><td>True       </td><td>()     </td><td>float64</td><td>1.0000000000000004e-05</td></tr>
</tbody>
</table>



To train the different models within the BO class.


```python
BO.train_models(iteration_Y=2500,iteration_C=2500)
```

    Training of the objective function model
    Training of constraint model 1
    ELBO: -73.6722504558447
    ELBO: -40.077754388823664
    ELBO: -36.954704944563375
    ELBO: -30.261831366775674
    ELBO: -25.24874356636096
    ELBO: -24.257270575123755
    ELBO: -19.260551011574375
    ELBO: -15.434903259765536
    ELBO: -12.238962616139355
    ELBO: -9.487753103811297
    ELBO: -7.649473460041234
    ELBO: -7.1675836827709904
    ELBO: -7.0369813742413445
    ELBO: -6.988259736356377
    ELBO: -6.970094851980103
    ELBO: -6.950827874353353
    ELBO: -6.920997679636944
    ELBO: -6.882030085583347
    ELBO: -6.834641777578899
    ELBO: -6.769723514374765
    ELBO: -6.724198240540412
    ELBO: -6.6658946241687556
    ELBO: -6.6057991675044345
    ELBO: -6.572771059572202
    ELBO: -6.5357604774463915
    ELBO: -6.502703875049761
    ELBO: -6.455007465227242
    ELBO: -6.457518841824695
    ELBO: -6.411861142049227
    ELBO: -6.420359098895663
    

### Infill criteria

The different infill criteria are implemented in "dgp_dace.Infill_criteria".
To define an unconstrained infill criterion, two arguments have to be given: y_min and the input dimension.


```python
BO.Ymin[-1]
```




    0.17355189315573133



Since the data used to train the model is normalized, Ymin has to be normalized before using it for the IC.


```python
IC = EI((BO.Ymin[-1]-BO.Y.mean(axis=0))/BO.Y.std(axis=0),BO.d) 
```

To evaluate the value of infill criterion we have to specify the model to use, the location x, how to evaluate the IC (analytic or using smapling)n and the number of samples.


```python
x= np.array([[0.5]])
IC.run(BO.model_Y,x,analytic=True)
```




    <tf.Tensor: shape=(1, 1), dtype=float64, numpy=array([[-2.62692319]])>




```python
IC.run(BO.model_Y,BO.X_n,analytic=False,num_samples=1000)
```




    <tf.Tensor: shape=(5, 1), dtype=float64, numpy=
    array([[-2.59701535e+00],
           [-2.34648074e+00],
           [-1.03330309e+00],
           [-2.36903727e+00],
           [-1.81666276e-03]])>



To optimize the IC the bounds have on which the optimization is done have to be specified, the method of optimization (Differential evolution or using Adam or DE+Adam) and their specific parameters.


```python
bounds = (BO.lw_n,BO.up_n)
IC.optimize(BO.model_Y,bounds,popsize_DE=300,popstd_DE = 1.5,iterations_DE=400, init_adam=None,iterations_adam=1000,method='DE+Adam',analytic=True)
```




    array([[0.15314149]])




```python
print(IC.x_opt) # The location of the IC optimum
```

    [[0.15314149]]
    

This is the normalized value. To get the orginal value we have to denormalize the location:


```python
print(IC.x_opt*BO.X.std(axis=0)+BO.X.mean(axis=0))
```

    [[0.50312589]]
    


```python
print(-IC.IC_optimized) # The value of the IC
```

    [[2.75201845]]
    

### Constrained IC

To define a constrained infill criterion such as the Expected violation we have to specify the array for which the different constraints are considered feasible. In SO_BO a constraint is considered feasible if it is inferior to zero. Thus, this array comes back to a zero array with a length equal to the number of constraint. However, since the constraint data are normalized, the array is not equal to zero anymore. To obtain the 'normalized zeros array' we do the following:


```python
BO.feasible_0
```




    array([-2.])



Hence, to define the EV: 


```python
IC_EV = EV(BO.feasible_0,BO.d)
```

To evaluate the value of infill criterion we have to specify the model to use, the location x, how to evaluate the IC (analytic or using smapling)n and the number of samples.


```python
x= np.array([[3.]])
IC_EV.run(BO.model_C,x,analytic=True)
```




    <tf.Tensor: shape=(1, 1), dtype=float64, numpy=array([[2.05025454]])>




```python
BO.up_n
```




    array([2.07156107])



To run the IC_EV along with the EI (or WB2, WB2S) we have to specify the treshold on which the EV is acceptable:


```python
IC_EV.run_with_IC(IC,BO.model_Y,BO.model_C,x,threshold=0.3,analytic=True,num_samples=100)
```




    <tf.Tensor: shape=(1, 1), dtype=float64, numpy=array([[10002.05025454]])>




```python
bounds = (BO.lw_n,BO.up_n)
```

For optimization:


```python
IC_EV.optimize_with_IC(IC,BO.model_Y,BO.model_C,bounds,threshold=0.1,analytic=True,num_samples=100,method='DE',popsize_DE=3000\
        ,popstd_DE = 1.5,iterations_DE=400, init_adam=None,iterations_adam=1000,)
```




    array([[0.5]])



This is the normalized value. To get the orginal value we have to denormalize the location:


```python
print(IC_EV.x_opt*BO.X.std(axis=0)+BO.X.mean(axis=0))
```

    [[0.47627208]]
    

### Run the BO loop


```python
BO.run(5,from_scratch=3,IC ='EI' ,train_iterations=4000,popsize_DE=300\
        ,popstd_DE = 3.,threshold=0.1,iterations_DE=400,constraint_handling='EV', init_adam=None,iterations_adam=1000,IC_method='DE+Adam',analytic=True)
```

    adding the most promising data point in iteration 0
    Training of the objective function model
    Training of constraint model 1
    ELBO: -73.6722504558447
    ELBO: -39.99402863857191
    ELBO: -36.95697133873597
    ELBO: -30.591088775681698
    ELBO: -25.248654139833324
    ELBO: -24.253462963639144
    ELBO: -19.255458872245647
    ELBO: -15.438522172113611
    ELBO: -12.242492818998645
    ELBO: -9.487940106951285
    ELBO: -7.6508259633750715
    ELBO: -7.165530583309681
    ELBO: -7.035753801289697
    ELBO: -6.992237579519968
    ELBO: -6.968615271787655
    ELBO: -6.949240726294715
    ELBO: -6.931270609718898
    ELBO: -6.891912568187555
    ELBO: -6.852125144898878
    ELBO: -6.79495883774212
    ELBO: -6.740401226669386
    ELBO: -6.68522955460525
    ELBO: -6.6176437047013446
    ELBO: -6.563475890598324
    ELBO: -6.535583132479516
    ELBO: -6.485925966314247
    ELBO: -6.443953214760138
    ELBO: -6.4432212304066105
    ELBO: -6.41168432486277
    ELBO: -6.400633095100558
    ELBO: -6.392131866276726
    ELBO: -6.380814383239343
    ELBO: -6.380727168503464
    ELBO: -6.33465376054593
    ELBO: -6.355368449805302
    ELBO: -6.343249590968402
    ELBO: -6.368957728862923
    ELBO: -6.297827536779819
    ELBO: -6.289314446164598
    ELBO: -6.337895971509448
    ELBO: -6.3077250645884355
    ELBO: -6.282237344611959
    ELBO: -6.331624520190595
    ELBO: -6.287201555384446
    ELBO: -6.333043822251824
    Actual Y min: 0.17355189315573133
    adding the most promising data point in iteration 1
    Training of the objective function model
    Training of constraint model 1
    ELBO: -73.87381316855146
    ELBO: -70.24431961912539
    ELBO: -69.25985901564825
    ELBO: -68.31277782662966
    ELBO: -67.44473134910518
    ELBO: -66.72638832449223
    ELBO: -59.614623442256296
    ELBO: -53.93211368781065
    ELBO: -48.639468836460274
    ELBO: -43.5056892332871
    ELBO: -38.45526864786287
    ELBO: -33.52210928018744
    ELBO: -28.753721137907423
    ELBO: -24.260384096626236
    ELBO: -20.068479200153508
    ELBO: -16.100466482481536
    ELBO: -12.397229902486558
    ELBO: -9.18342535873448
    ELBO: -6.858574207919242
    ELBO: -5.656805277878288
    ELBO: -5.263119574829381
    ELBO: -5.174033905885542
    ELBO: -5.149011112123164
    ELBO: -5.138732218017408
    ELBO: -5.137317298583882
    Actual Y min: 0.17355189315573133
    adding the most promising data point in iteration 2
    Training of the objective function model
    Training of constraint model 1
    ELBO: -71.50469068758217
    ELBO: -70.04392304264834
    ELBO: -70.04950793797646
    ELBO: -70.04090658549553
    ELBO: -70.04660127183908
    ELBO: -70.03934810002752
    ELBO: -63.759585726487025
    ELBO: -58.40153387088199
    ELBO: -53.29918599802801
    ELBO: -48.21863679756178
    ELBO: -43.17745181711393
    ELBO: -38.150258973878294
    ELBO: -33.12030347186266
    ELBO: -28.110106039287345
    ELBO: -23.13543228401987
    ELBO: -18.251038632573845
    ELBO: -13.582782462584742
    ELBO: -9.486062390480688
    ELBO: -6.45747232355083
    ELBO: -4.832396472335561
    ELBO: -4.303909169509138
    ELBO: -4.166603693774185
    ELBO: -4.1328227547463
    ELBO: -4.133050494703493
    ELBO: -4.147069592357839
    Actual Y min: 0.17355189315573133
    adding the most promising data point in iteration 3
    The DGP architecture
    layer 1 : dim_in 1 --> dim_out 1
    layer 2 : dim_in 1 --> dim_out 1
    Training of the objective function model
    Training of constraint model 1
    ELBO: -117.87560072944773
    ELBO: -87.70102474223837
    ELBO: -80.15712316264155
    ELBO: -64.75867123113896
    ELBO: -62.878291889081886
    ELBO: -61.83145941514723
    ELBO: -52.24199002625777
    ELBO: -44.35337362754089
    ELBO: -37.49093810403607
    ELBO: -30.104060939693564
    ELBO: -25.31560493133661
    ELBO: -20.4569064029047
    ELBO: -16.752068880848128
    ELBO: -13.687200058421244
    ELBO: -10.844950279190579
    ELBO: -9.0377843583225
    ELBO: -7.106139949769043
    ELBO: -4.469020849410114
    ELBO: -2.9327029813874272
    ELBO: -2.0122653426709487
    ELBO: -1.3612572173165223
    ELBO: -0.7913820819499158
    ELBO: -0.6343472634893175
    ELBO: -0.47744267750505287
    ELBO: -0.5216134885424921
    ELBO: -0.04071344043952507
    ELBO: -0.013011246767646867
    ELBO: 0.06518272966339822
    ELBO: 0.10952144687989929
    ELBO: 0.12081832816318183
    ELBO: 0.2572391603452111
    ELBO: 0.25466331635611894
    ELBO: 0.37576863466318855
    ELBO: 0.37399256732620856
    ELBO: 0.5018438355824131
    ELBO: 0.5356745548140438
    ELBO: 0.4151090444761891
    ELBO: 0.43052210885399944
    ELBO: 0.3867817142741572
    ELBO: 0.575768360178607
    ELBO: 0.6381778564334724
    ELBO: 0.4339181246986996
    ELBO: 0.6793839682163849
    ELBO: 0.6384477310430441
    ELBO: 0.6703552820536984
    Actual Y min: 0.15076040083174433
    adding the most promising data point in iteration 4
    Training of the objective function model
    Training of constraint model 1
    ELBO: -505.2406525576146
    ELBO: -135.06577363651334
    ELBO: -119.82157330790734
    ELBO: -112.05886301238114
    ELBO: -110.1234237963303
    ELBO: -109.85037755054307
    ELBO: -99.33616410456968
    ELBO: -89.26891222766996
    ELBO: -79.46602633659249
    ELBO: -70.13128860476877
    ELBO: -61.30297290191638
    ELBO: -52.853107790155924
    ELBO: -44.67675384194179
    ELBO: -36.60106512209427
    ELBO: -28.926201733805502
    ELBO: -21.479757092694783
    ELBO: -14.786383212682125
    ELBO: -9.094617235579635
    ELBO: -5.042477796338201
    ELBO: -2.950245964435217
    ELBO: -2.222747207687936
    ELBO: -2.0950172700164096
    ELBO: -2.0578522594528614
    ELBO: -1.923632009576755
    ELBO: -1.9100514212037787
    Actual Y min: 0.142512394873708
    


```python
BO.Ymin
```




    array([0.17355189, 0.17355189, 0.17355189, 0.17355189, 0.1507604 ,
           0.14251239])




```python
BO.run(5,from_scratch=3,IC ='EI' ,train_iterations=4000,popsize_DE=300\
        ,popstd_DE = 3.,threshold=0.1,iterations_DE=400,constraint_handling='EV', init_adam=None,iterations_adam=1000,IC_method='DE+Adam',analytic=True)
```

    adding the most promising data point in iteration 0
    Training of the objective function model
    Training of constraint model 1
    ELBO: -104.36575738627668
    ELBO: -104.38475212909161
    ELBO: -104.32474319308939
    ELBO: -104.35587327943247
    ELBO: -104.39945629491378
    ELBO: -104.3089025221638
    ELBO: -96.27762583693897
    ELBO: -88.16540087401101
    ELBO: -80.06818210429904
    ELBO: -71.99747610590587
    ELBO: -63.93136082897597
    ELBO: -55.90118194505523
    ELBO: -47.793592997824085
    ELBO: -39.7944138929898
    ELBO: -31.768202518638702
    ELBO: -24.007248229046404
    ELBO: -16.557884955758226
    ELBO: -10.015458409216624
    ELBO: -5.103881230493567
    ELBO: -2.5264697072848072
    ELBO: -1.6440034439430296
    ELBO: -1.4732316004250166
    ELBO: -1.4752416682635534
    ELBO: -1.6111434731714738
    ELBO: -1.444617358263116
    ELBO: -1.4218482618600614
    ELBO: -1.4446377486217337
    ELBO: -1.419944633490985
    ELBO: -1.549706725633241
    ELBO: -1.6804592931055158
    ELBO: -1.4528856177170635
    ELBO: -1.545354161421212
    ELBO: -1.452487148515317
    ELBO: -1.4858893741262555
    ELBO: -1.5055701983107497
    ELBO: -1.4088947843665807
    ELBO: -1.4612305459863464
    ELBO: -1.4489632687865388
    ELBO: -1.4543817172526161
    ELBO: -1.4026619016243966
    ELBO: -1.5854439499662067
    ELBO: -1.4300918577821413
    ELBO: -1.4456522208956564
    ELBO: -1.4738716872562598
    ELBO: -1.4766115395823132
    Actual Y min: 0.13964761060813022
    adding the most promising data point in iteration 1
    Training of the objective function model
    Training of constraint model 1
    ELBO: -224.83102570122918
    ELBO: -115.11705175692252
    ELBO: -111.98130539709439
    ELBO: -111.86583832848221
    ELBO: -111.9744366216067
    ELBO: -111.86492951000514
    ELBO: -100.49213396087073
    ELBO: -89.31874589496731
    ELBO: -79.68104734235885
    ELBO: -71.24255801346393
    ELBO: -62.990242324815185
    ELBO: -54.971880124171236
    ELBO: -46.79976007138663
    ELBO: -38.66947355981816
    ELBO: -30.610332948711477
    ELBO: -22.757607911833446
    ELBO: -15.334811892217967
    ELBO: -8.721412237017546
    ELBO: -3.7915186117964197
    ELBO: -1.198134248034524
    ELBO: -0.30723203248745534
    ELBO: 0.010957422256559823
    ELBO: 0.10887841544298027
    ELBO: 0.05516252287069534
    ELBO: 0.2202632596513432
    Actual Y min: 0.13033501920696328
    adding the most promising data point in iteration 2
    Training of the objective function model
    Training of constraint model 1
    ELBO: -125.42014444044524
    ELBO: -104.7551765790117
    ELBO: -104.76582196417776
    ELBO: -104.7035864136836
    ELBO: -104.88440925310596
    ELBO: -104.6541745472323
    ELBO: -95.01395688540845
    ELBO: -86.46645213601357
    ELBO: -78.26698791983544
    ELBO: -70.12520694871291
    ELBO: -62.04948175216232
    ELBO: -53.94454764278451
    ELBO: -45.92034797997374
    ELBO: -37.83578630457339
    ELBO: -29.73571680473427
    ELBO: -21.888180744173578
    ELBO: -14.469188845389368
    ELBO: -7.72707938888205
    ELBO: -2.8514389063074894
    ELBO: -0.13740195053576088
    ELBO: 0.6915681186232483
    ELBO: 0.9229862520040761
    ELBO: 1.0767454322285168
    ELBO: 1.0789979697489152
    ELBO: 1.139787915159392
    Actual Y min: 0.12322374483022482
    adding the most promising data point in iteration 3
    The DGP architecture
    layer 1 : dim_in 1 --> dim_out 1
    layer 2 : dim_in 1 --> dim_out 1
    Training of the objective function model
    Training of constraint model 1
    ELBO: -191.54785118534446
    ELBO: -155.67870803340134
    ELBO: -148.07336831017486
    ELBO: -136.14747106592932
    ELBO: -129.493130747882
    ELBO: -128.11287125381534
    ELBO: -109.51616864961144
    ELBO: -96.01950549879663
    ELBO: -83.80152173922251
    ELBO: -71.18357636052954
    ELBO: -57.596823532590626
    ELBO: -47.906640499557625
    ELBO: -40.60773333896189
    ELBO: -33.68403055796088
    ELBO: -25.18791521343994
    ELBO: -17.37223532928291
    ELBO: -10.02896602074133
    ELBO: -3.5012442750232466
    ELBO: 1.2731209717361551
    ELBO: 4.4001660978608825
    ELBO: 6.402163098802074
    ELBO: 6.712734030982094
    ELBO: 7.308464737379872
    ELBO: 7.491520977104109
    ELBO: 7.425566158617514
    ELBO: 7.884344140293663
    ELBO: 8.390687530399127
    ELBO: 8.49030065919326
    ELBO: 8.516281194599905
    ELBO: 8.436482807036427
    ELBO: 8.823115442931062
    ELBO: 8.801488517668538
    ELBO: 8.80587989598012
    ELBO: 8.51773242913788
    ELBO: 9.565407292909182
    ELBO: 9.217666443126852
    ELBO: 9.587790387161142
    ELBO: 9.017520059736178
    ELBO: 9.707965510008556
    ELBO: 7.566045688510087
    ELBO: 9.85465382108763
    ELBO: 9.577395951835797
    ELBO: 10.232521042887385
    ELBO: 9.477423036615981
    ELBO: 10.31651524222076
    Actual Y min: 0.09982584728917254
    adding the most promising data point in iteration 4
    Training of the objective function model
    Training of constraint model 1
    ELBO: -1112.3319548128607
    ELBO: -191.58669498486168
    ELBO: -176.8821924561114
    ELBO: -173.64726851857415
    ELBO: -172.51025404993788
    ELBO: -172.56660663333406
    ELBO: -153.8352152494104
    ELBO: -137.72892466271904
    ELBO: -121.40878722936421
    ELBO: -106.00170837214799
    ELBO: -91.45395729212267
    ELBO: -78.34022208437435
    ELBO: -64.5975694022934
    ELBO: -51.6704302809506
    ELBO: -39.85215882879696
    ELBO: -26.0817072893023
    ELBO: -13.692767926138245
    ELBO: -3.29662216345298
    ELBO: 4.246256223111452
    ELBO: 7.563912292472324
    ELBO: 9.338641271723347
    ELBO: 8.696461731113985
    ELBO: 9.395981430112904
    ELBO: 9.13692359264003
    ELBO: 4.764993055509727
    Actual Y min: 0.08509663947575162
    


```python
BO.run(3,from_scratch=3,IC ='EI' ,train_iterations=4000,popsize_DE=300\
        ,popstd_DE = 3.,threshold=0.1,iterations_DE=400,constraint_handling='EV', init_adam=None,iterations_adam=1000,IC_method='DE+Adam',analytic=True)
```

    adding the most promising data point in iteration 0
    Training of the objective function model
    Training of constraint model 1
    ELBO: -157.60904236881004
    ELBO: -157.11634931201428
    ELBO: -158.20926569026386
    ELBO: -158.10546979949444
    ELBO: -156.98337250741815
    ELBO: -157.1446827767565
    ELBO: -144.66713600823994
    ELBO: -132.24192326032625
    ELBO: -117.46913936212586
    ELBO: -104.64350809024413
    ELBO: -91.07781658276801
    ELBO: -78.37537671353752
    ELBO: -65.03653094098769
    ELBO: -51.78080051178112
    ELBO: -39.97766805921031
    ELBO: -26.91828659940508
    ELBO: -14.520027243783872
    ELBO: -3.7594355987326864
    ELBO: 3.7296486752884377
    ELBO: 8.573322886430994
    ELBO: 9.78814576620423
    ELBO: 10.364282941588606
    ELBO: 9.216877838908374
    ELBO: 10.019277823929961
    ELBO: 9.089663447747952
    ELBO: 10.097469141865616
    ELBO: 9.114295119034814
    ELBO: 10.625638767149034
    ELBO: 8.96048749375204
    ELBO: 10.379507737482555
    ELBO: 10.227030740254925
    ELBO: 10.37375168655069
    ELBO: 10.389201763903394
    ELBO: 10.084100964675052
    ELBO: 10.123333332360218
    ELBO: 10.345154015827816
    ELBO: 9.052413703550798
    ELBO: 9.852797547713699
    ELBO: 10.189259551438198
    ELBO: 9.7234213226264
    ELBO: 10.659481580461154
    ELBO: 10.826540455704517
    ELBO: 10.898544584409294
    ELBO: 9.933037387562102
    ELBO: 10.488858535654849
    Actual Y min: 0.08210944857604378
    adding the most promising data point in iteration 1
    Training of the objective function model
    Training of constraint model 1
    ELBO: -3807.16642765495
    ELBO: -248.28894239127663
    ELBO: -200.8083513886552
    ELBO: -183.78250256385172
    ELBO: -179.96875404879393
    ELBO: -179.5619632421573
    ELBO: -162.52373134750252
    ELBO: -148.17395266119945
    ELBO: -129.6668552109385
    ELBO: -114.48468112654102
    ELBO: -98.46947479388749
    ELBO: -82.60270964768144
    ELBO: -68.15257484943533
    ELBO: -54.971155094580695
    ELBO: -41.22854493968248
    ELBO: -28.104972242696306
    ELBO: -15.378177147826605
    ELBO: -5.052355240934332
    ELBO: 1.8322395130051987
    ELBO: 8.142970581183476
    ELBO: 7.856682876908387
    ELBO: 10.713662993004661
    ELBO: 10.966558413278314
    ELBO: 11.242962131135634
    ELBO: 10.836711665119267
    Actual Y min: 0.07102298226839295
    adding the most promising data point in iteration 2
    Training of the objective function model
    Training of constraint model 1
    ELBO: -698.2465504414002
    ELBO: -183.80653441762576
    ELBO: -173.07196812822116
    ELBO: -171.5655985340524
    ELBO: -171.9371091917217
    ELBO: -172.025076084177
    ELBO: -154.49172069895727
    ELBO: -138.3178294618598
    ELBO: -121.43531478412291
    ELBO: -106.217339824106
    ELBO: -92.93169494800446
    ELBO: -79.93212883662069
    ELBO: -67.25015357452823
    ELBO: -53.92370024995972
    ELBO: -40.56877860923623
    ELBO: -28.342926477700495
    ELBO: -15.367874398550718
    ELBO: -4.730873001963609
    ELBO: 3.54316964889653
    ELBO: 7.968882329159364
    ELBO: 9.457342159075253
    ELBO: 8.928275335030115
    ELBO: 9.961243573314157
    ELBO: 7.537066517630869
    ELBO: 9.92399777075316
    Actual Y min: 0.06255951879737233
    


```python
BO.Ymin
```




    array([0.17355189, 0.17355189, 0.17355189, 0.17355189, 0.1507604 ,
           0.14251239, 0.13964761, 0.13033502, 0.12322374, 0.09982585,
           0.08509664, 0.08210945, 0.07102298, 0.06255952])




```python
plt.figure()
plt.plot(np.arange(0,len(BO.Ymin),1),BO.Ymin)
```




    [<matplotlib.lines.Line2D at 0x7f3f482450b8>]




![png](https://hebbalali.github.io/Hebbalali/assets/output_66_1.png)


### Now using a GP for the constraint


```python
model_C_dic = {'num_layers':0,'kernels':'rbf'}
BO_GP = SO_BO(problem,DoE_size=5,model_Y_dic=model_Y_dic,model_C_dic=model_C_dic,normalize_input=True)

```


```python
BO_GP.run(5,from_scratch=3,IC ='EI' ,train_iterations=2000,popsize_DE=300\
        ,popstd_DE = 3.,threshold=0.1,iterations_DE=400,constraint_handling='EV', init_adam=None,iterations_adam=1000,IC_method='DE+Adam',analytic=True)
BO_GP.run(3,from_scratch=3,IC ='EI' ,train_iterations=2000,popsize_DE=300\
        ,popstd_DE = 3.,threshold=0.1,iterations_DE=400,constraint_handling='EV', init_adam=None,iterations_adam=1000,IC_method='DE+Adam',analytic=True)
BO_GP.run(5,from_scratch=3,IC ='EI' ,train_iterations=2000,popsize_DE=300\
        ,popstd_DE = 3.,threshold=0.1,iterations_DE=400,constraint_handling='EV', init_adam=None,iterations_adam=1000,IC_method='DE+Adam',analytic=True)
```

    adding the most promising data point in iteration 0
    Training of the objective function model
    Training of constraint model 1
    Actual Y min: 0.1616389283660566
    adding the most promising data point in iteration 1
    Training of the objective function model
    Training of constraint model 1
    Actual Y min: 0.13687942764692898
    adding the most promising data point in iteration 2
    Training of the objective function model
    Training of constraint model 1
    Actual Y min: 0.10711606267747553
    adding the most promising data point in iteration 3
    Training of the objective function model
    Training of constraint model 1
    Actual Y min: 0.09206730846372368
    adding the most promising data point in iteration 4
    Training of the objective function model
    Training of constraint model 1
    Actual Y min: 0.07553200757387137
    adding the most promising data point in iteration 0
    Training of the objective function model
    Training of constraint model 1
    Actual Y min: 0.07346148679562055
    adding the most promising data point in iteration 1
    Training of the objective function model
    Training of constraint model 1
    Actual Y min: 0.07346148679562055
    adding the most promising data point in iteration 2
    Training of the objective function model
    Training of constraint model 1
    Actual Y min: 0.07346148679562055
    adding the most promising data point in iteration 0
    Training of the objective function model
    Training of constraint model 1
    Actual Y min: 0.0729086900853031
    adding the most promising data point in iteration 1
    Training of the objective function model
    Training of constraint model 1
    Actual Y min: 0.0706684120790352
    adding the most promising data point in iteration 2
    Training of the objective function model
    Training of constraint model 1
    Actual Y min: 0.06888162934222057
    adding the most promising data point in iteration 3
    Training of the objective function model
    Training of constraint model 1
    Actual Y min: 0.06888162934222057
    adding the most promising data point in iteration 4
    Training of the objective function model
    Training of constraint model 1
    Actual Y min: 0.06888162934222057
    


```python
plt.figure()
plt.plot(np.arange(0,len(BO.Ymin),1),BO.Ymin,label='BO_DGP')
plt.plot(np.arange(0,len(BO.Ymin),1),BO_GP.Ymin,label='BO-GP')
plt.legend()
```




    <matplotlib.legend.Legend at 0x7f3ee4600f28>




![png](https://hebbalali.github.io/Hebbalali/assets/output_70_1.png)

