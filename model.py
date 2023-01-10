from config import *
class ToxicModel(nn.Module):
    def __init__(self, model_name):
        super(ToxicModel, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.drop = nn.Dropout(p=0.3)
        self.fc = nn.Linear(768, CONFIG['num_classes'])
        
    def forward(self, comment_ids, comment_mask, token_type_ids):        
        x = self.model(input_ids=comment_ids,attention_mask=comment_mask,output_hidden_states=False)
        x = self.drop(x[0][:,0])
        x = self.fc(x)
        outputs = x
        return outputs