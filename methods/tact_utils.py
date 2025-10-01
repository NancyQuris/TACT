import torch 
import torchvision.transforms.v2 as v2
import random
from transformers import DistilBertTokenizerFast

from datasets.civil import initialize_bert_transform

class CivilAugmentation(object):
    def __init__(self, args):
        self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        self.transform = initialize_bert_transform(args, args_dict=True)
        self.templates = [
            'This is a post about females and males.',
            'The discussion focuses on women and men.',
            'Females and males are the central topic here.',
            'Women and men both contribute to this conversation.',
            'This explores perspectives of females and males.',
            'The post highlights contributions of women and men.',
            'Both females and males are part of the narrative.',
            'Women and men play essential roles in this story.',
            'Females and males are equally represented here.',
            'This covers aspects of both women and men.',
            'This is a post about women and men.',
            'The discussion centers on ladies and gentlemen.',
            'Females and males are the key focus here.',
            'Girls and boys both play significant roles.',
            'Both genders are part of this discussion.',
            'This highlights contributions from men and women.',
            'Ladies and gentlemen are represented here equally.',
            'The focus is on both sexes and their roles.',
            'Womenfolk and menfolk shape this narrative.',
            'Both males and females are included in this topic.',
            'This is a post about LGBTQ+ and heterosexual individuals.',
            'The discussion focuses on sexual minorities and heterosexual communities.',
            'This highlights experiences of both LGBTQ+ and cisgender people.',
            'The post compares queer and non-queer perspectives.',
            'This covers topics relevant to both LGBTQ+ and straight groups.',
            'Gender-diverse and cisgender voices are included in this conversation.',
            'The focus is on LGBTQ+ and heterosexual rights and issues.',
            'Both sexual minorities and heterosexual people’s experiences are addressed here.',
            'This post examines the lives of gender-nonconforming and cisgender individuals.',
            'The post explores the intersection of queer and non-queer identities.',
            'LGBTQ+ and heterosexual people both contribute to this topic.',
            'This content engages with both gender-diverse and cisgender communities.',
            'The article offers insights into the experiences of LGBTQ+ and non-LGBTQ+ individuals.',
            'This is a post about LGBTQ+ and heterosexual experiences in society.',
            'Both sexual minorities and heterosexual groups have a place in this discussion.',
            'This conversation includes both LGBTQ+ and cisgender perspectives.',
            'We explore issues affecting both sexual minorities and heterosexual individuals.',
            'This is about the relationships between LGBTQ+ and heterosexual people.',
            'The focus is on creating unity between LGBTQ+ and cisgender communities.',
            'This post discusses challenges faced by both gender-diverse and cisgender people.',
            'This is a post about Christians, Muslims, and followers of other faiths.',
            'The discussion focuses on Christians, Muslims, and practitioners of different religions.',
            'This highlights the experiences of Christians, Muslims, and believers from various traditions.',
            'The post compares Christian, Muslim, and other spiritual practices.',
            'This covers topics relevant to Christians, Muslims, and people of other religious backgrounds.',
            'The voices of Christians, Muslims, and adherents of different faiths are included in this conversation.',
            'The focus is on Christian, Muslim, and interfaith perspectives.',
            'Both Christians, Muslims, and people of other beliefs contribute to this discussion.',
            'This post examines the lives of Christians, Muslims, and followers of other religions.',
            'The post explores the intersection of Christianity, Islam, and other spiritual practices.',
            'Christians, Muslims, and people from diverse faiths share common values of compassion.',
            'This content engages with Christians, Muslims, and those from various religious traditions.',
            'The article offers insights into the teachings of Christians, Muslims, and other faith communities.',
            'This is a post about Christians, Muslims, and adherents of various world religions.',
            'Both Christians, Muslims, and individuals from different belief systems are included in this conversation.',
            'The focus is on how Christians, Muslims, and people of other religions practice faith.',
            'This conversation includes insights from Christians, Muslims, and followers of other spiritual paths.',
            'We’ll explore issues affecting Christians, Muslims, and people from various religious backgrounds.',
            'This is about the relationships between Christians, Muslims, and those of other beliefs.',
            'The post discusses shared values between Christians, Muslims, and adherents of other religions.',
            'This is a post about Black and White communities.',
            'The discussion focuses on African American and Caucasian experiences.',
            'This highlights the perspectives of Black and White individuals.',
            'The post compares the lives of Black and White people.',
            'This covers topics relevant to both Black and White races.',
            'The voices of African Americans and Caucasians are included in this conversation.',
            'The focus is on Black and White racial dynamics.',
            'Both Black and White communities contribute to this discussion.',
            'This post examines the experiences of Black and White individuals.',
            'The post explores the intersection of African American and European American identities.',
            'Black and White people play vital roles in shaping society.',
            'This content engages with the experiences of Black and White groups.',
            'The article offers insights into the lives of Black and White people in different settings.',
            'This is a post about African American and White American experiences.',
            'Both Black and White cultures have unique contributions to the world.',
            'The focus is on both Black and White perspectives in social issues.',
            'This conversation includes both Black and White voices.',
            'We’ll explore the relationship between Black and White individuals.',
            'This is about the interactions between African Americans and Caucasians.',
            'The post discusses challenges faced by both Black and White communities.',
        ]

    def __call__(self, token):
        # decode embedding to text
        text_input = self.tokenizer.decode(
            token[:, 0],
            skip_special_tokens=True, 
            clean_up_tokenization_sapaces=True
        )
        
        template = random.choice(self.templates)
        if random.random() < 0.5:
            text_input = template + ' ' + text_input
        else:
            text_input = text_input + ' ' + template
        
        return self.transform(text_input)

class StainColorJitter(object):
    M = torch.tensor([
        [0.65, 0.70, 0.29],
        [0.07, 0.99, 0.11],
        [0.27, 0.57, 0.78]])
    Minv = torch.inverse(M)
    eps = 1e-6

    def __init__(self, device, sigma=0.05):
        # Sigma specifies the strength of the augmentation
        self.sigma = sigma
        self.device = device 

    def __call__(self, P):
        # Expects P to be the result of ToTensor, i.e., each pixel in [0, 1]
        # IPython.embed()
        assert P.shape == (3, 96, 96)
        assert torch.max(P) <= 1.0
        assert torch.min(P) >= 0.0

        # Eqn 5
        S = - (torch.log(255 * P.T + self.eps)).matmul(self.Minv.to(self.device)) # 96 x 96 x 3

        # alpha is uniform from U(1 - sigma, 1 + sigma)
        alpha = 1 + (torch.rand(3) - 0.5) * 2 * self.sigma

        # beta is uniform from U(-sigma, sigma)
        beta = (torch.rand(3) - 0.5) * 2 * self.sigma

        # Eqn 6
        Sp = S * alpha.to(self.device) + beta.to(self.device)

        # Eqn 7
        Pp = torch.exp(-Sp.matmul(self.M.to(self.device))) - self.eps

        # Transpose, rescale, and clip
        Pp = Pp.T / 255
        Pp = torch.clip(Pp, 0.0, 1.0)

        return Pp

def get_augmentation(args, device):
    if args['dataset'] == 'camelyon':
        base_augmentation = v2.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        stain_aug = StainColorJitter(device=device)
        non_causal_augmentation = v2.Compose([
            stain_aug,
            v2.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
        
    elif args['dataset'] == 'birdcalls':
        base_augmentation = v2.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        jitter_aug = v2.ColorJitter(brightness=0, contrast=0, saturation=2,hue=0.5)
        non_causal_augmentation = v2.Compose([
            jitter_aug,
            v2.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
        
    elif args['dataset'] == 'civil':
        base_augmentation = lambda x: x
        non_causal_augmentation = CivilAugmentation(args)

    elif args['dataset'] in ['imagenet_r', 'imagenetv2']:
        base_augmentation = v2.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        if args['dataset'] == 'imagenet_r':
            non_causal_augmentation = v2.Compose([
                    v2.ToDtype(torch.uint8, scale=True),
                    v2.RandAugment(),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                ])
        elif args['dataset'] == 'imagenetv2':
            non_causal_augmentation = v2.Compose([
                    v2.ToDtype(torch.uint8, scale=True),
                    v2.AutoAugment(),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                ])
        
    return base_augmentation, non_causal_augmentation

def get_PCs(X):
    mean = X.mean(dim=1, keepdim=True)
    X_mean = X - mean 
    XCov = torch.bmm(X_mean.transpose(1, 2), X_mean)
    try: 
        # for PSD symmetric matrices, SVD and eigendecomposition give us the same result 
        _, eigenvalues, eigenvectors = torch.linalg.svd(XCov, full_matrices=False) # eigenvalues are ordered in descending order
    except torch._C._LinAlgError as e:
        print(f"LinAlgError encountered: {e}")
        eigenvalues, eigenvectors = None, None
    return eigenvalues, eigenvectors, mean

def projection(vector_to_project, project_direction):
    ''' the project direction should be a unit vector '''
    # project a to b = (a dot b / ||b||^2)[the magnitude, ||b|| is b's magnitude] b [the direction]
    project_direction = project_direction.unsqueeze(-2) if vector_to_project.dim() == 3 else project_direction
    # Compute dot product along the last dimension (D)
    dot_product = (vector_to_project * project_direction).sum(dim=-1, keepdim=True)
    # Compute projected vector
    vp_projected = dot_product * project_direction
    return vp_projected

def remove_PCs(features, PCs, start_pc, num_of_pc_to_remove):
    for i in range(num_of_pc_to_remove):
        features = features - projection(features, PCs[:, i+start_pc, :]) # get the row vectors 
    return features 
    