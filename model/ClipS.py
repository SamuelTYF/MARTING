import torch
import math

class Co_attention(torch.nn.Module):
    def __init__(self,len):
        super().__init__()
        self.len=len
        weights = torch.Tensor(len, len)
        self.weights = torch.nn.Parameter(weights)
        torch.nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        self.tanh = torch.nn.Tanh()

    def forward(self, H, T):
        batch=H.shape[0]
        T_tmp = T.clone()
        T = T.permute(0, 2, 1)
        z_1 = torch.matmul(H, self.weights)
        z_2 = torch.matmul(z_1, T)
        C = self.tanh(z_2)
        # 768
        HT = torch.matmul(C, T_tmp).reshape(batch, self.len)
        return HT

class ClipS(torch.nn.Module):
    def __init__(self,clip,opt):
        super(ClipS, self).__init__()
        self.clip=clip.float()
        assert("len" in opt)
        self.len=opt["len"]
        self.fc = torch.nn.Linear(self.len+self.len,2)
        self.co_attention = Co_attention(self.len)
        self.sigm = torch.nn.Sigmoid()
        assert("input1" in opt)
        assert("input2" in opt)
        self.input1=opt["input1"]
        self.input2=opt["input2"]

    def forward(self,input):

        text_features = self.clip.encode_text(input[self.input1])
        image_features = self.clip.encode_image(input[self.input2])

        o_features=text_features- self.co_attention(image_features.unsqueeze(1),text_features.unsqueeze(1))

        out = torch.cat((text_features, o_features), dim = 1)
        out = self.fc(out)
        result = self.sigm(out)

        del text_features, image_features, o_features, out

        return result

def build_ClipS(opt,requirements):
    assert("clip" in requirements)
    return ClipS(requirements["clip"],opt)