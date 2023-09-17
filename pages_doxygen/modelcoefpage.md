# Model coeficients
Here in this folder you can find the last and best coeficients obtained during training for each stage. To load these coeficients in the following way
```python
>>> import MSECNN
>>> import train_model_utils
>>>
>>> # Initialize parameters
>>> path_to_folder_with_model_params = "model_coefficients/best_coefficients"
>>> device = "cuda:0"
>>> qp = 32  # Quantisation Parameter
>>> 
>>> # Initialize Model
>>> stg1_2 = msecnn.MseCnnStg1(device=device, QP=qp).to(device)
>>> stg3 = msecnn.MseCnnStgX(device=device, QP=qp).to(device)
>>> stg4 = msecnn.MseCnnStgX(device=device, QP=qp).to(device)
>>> stg5 = msecnn.MseCnnStgX(device=device, QP=qp).to(device)
>>> stg6 = msecnn.MseCnnStgX(device=device, QP=qp).to(device)
>>> model = (stg1_2, stg3, stg4, stg5, stg6)
>>>
>>> model = train_model_utils.load_model_parameters_eval(model, path_to_folder_with_model_params, device)  # Load for all of the stages
```
