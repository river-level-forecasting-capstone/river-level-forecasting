#%%
from aws_dispatcher import AWSDispatcher

dispatcher = AWSDispatcher('illinois-kerby', 'Block_GRU_6hour', 'Linear Regression')

dispatcher.rebuild_current_forecast(update_dataset=True)
# %%