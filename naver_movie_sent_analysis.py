
# coding: utf-8

# In[2]:


train_data_path = '../../data/nsmc/ratings_train.txt'
test_data_path = '../../data/nsmc/ratings_test.txt'

def read_data(filename):
    with open(filename, 'r', encoding='UTF-8') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
        # txt 파일의 헤더(id document label)는 제외하기
        data = data[1:]
    return data

train_data = read_data(train_data_path)
test_data = read_data(test_data_path)


# In[3]:


train_data[:5]


# In[4]:


for row in train_data:
    if not row[0] or not row[1] or not row[2]:
        print(row)


# In[5]:


error_indices = []
for i, row in enumerate(train_data):
    if not row[1]:
        error_indices.append(i)


# In[6]:


for row in train_data:
    if not row[1]:
        train_data.remove(row)


# In[7]:


for row in test_data:
    if not row[1]:
        test_data.remove(row)


# In[8]:


from khaiii import KhaiiiApi


# In[9]:


api = KhaiiiApi()


# In[10]:


train_pos = []
for row in train_data:
    sent_pos = []
    sentence = row[1]
    for word in api.analyze(sentence):
        pos = str(word).split('\t')[1]
        sent_pos.append(pos)
    train_pos.append(sent_pos)
train_pos[:5]


# In[11]:


test_pos = []
for row in test_data:
    sent_pos = []
    sentence = row[1]
    for word in api.analyze(sentence):
        pos = str(word).split('\t')[1]
        sent_pos.append(pos)
    test_pos.append(sent_pos)
test_pos[:5]


# In[12]:


print(len(train_pos))
print(len(test_pos))


# In[13]:


import nltk


# In[14]:


tokens = [token for tokens_in_sentence in train_pos for token in tokens_in_sentence]


# In[15]:


len(tokens)


# In[16]:


text = nltk.Text(tokens, name='NMSC')


# In[17]:


print(len(set(text.tokens)))


# In[18]:


from pprint import pprint

pprint(text.vocab().most_common(10))


# In[19]:


import matplotlib.pyplot as plt
from matplotlib import rc

plt.rcParams["font.family"] = 'NanumGothicCoding'
plt.rcParams["font.size"] = 8
plt.rcParams["figure.figsize"] = (14,4)

plt.figure(figsize=(20,10))
text.plot(50)


# In[20]:


selected_words = [f[0] for f in text.vocab().most_common(10000)]


# In[21]:


train_docs = [(pos, row[2]) for row, pos in zip(train_data, train_pos)]
test_docs = [(pos, row[2]) for row, pos in zip(train_data, test_pos)]


# In[22]:


import six
six.MAXSIZE


# In[23]:


def term_frequency(doc):
    result = []
    for word in selected_words:
        result.append(doc.count(word))
    return result


# In[27]:


def memory():
    """
    Get node total memory and memory usage
    """
    with open('/proc/meminfo', 'r') as mem:
        ret = {}
        tmp = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) == 'MemTotal:':
                ret['total'] = int(sline[1])
            elif str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                tmp += int(sline[1])
        ret['free'] = tmp
        ret['used'] = int(ret['total']) - int(ret['free'])
    return ret


# In[28]:


memory()


# In[ ]:


train_x = []
for d, _ in train_docs:
    train_x.append(term_frequency(d))


# In[ ]:


train_x = [term_frequency(d) for d, _ in train_docs]


# In[ ]:


test_x = [term_frequency(d) for d, _ in test_docs]
train_y = [c for _, c in train_docs]
test_y = [c for _, c in test_docs]


# In[19]:


i = 0
ls = []
for i, value in enumerate(train_x[0]):
    if value == 1:
        ls.append(i)
    i += 1


# In[20]:


ls


# In[17]:


pprint(train_x[0])


# In[18]:


pprint(train_y[0])


# In[ ]:


import numpy as np

x_train = np.asarray(train_x).astype('int64')
x_test = np.asarray(test_x).astype('int64')

y_train = np.asarray(train_y).astype('int64')
y_test = np.asarray(test_y).astype('int64')


# In[ ]:


from torch.utils.data import Dataset, DataLoader


# In[ ]:


import torch


# In[ ]:


class PosDataset(Dataset):
    def __init__(self, x_train, y_train):
        x_yet = torch.from_numpy(x_train)
        self.x_train = x_yet.float()
        y_yet = torch.from_numpy(y_train)
        self.y_train = y_yet.float().view(-1,1)
    
    def __len__(self):
        return len(self.y_train)
    
    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]


# In[ ]:


trainset = PosDataset(x_train, y_train)
trainset_loader = DataLoader(trainset, batch_size=128)


# In[ ]:


import torch.nn as nn
import torch.nn.functional as F


# In[ ]:


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(256, 64) 
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.sigmoid(x)
        return x

net = Net()


# In[ ]:


import torch.optim as optim

criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


# In[ ]:


for epoch in range(1000):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainset_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        x = data[0]
        y = data[1].view(-1,1)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 1000 mini-batches
            print('[%d, %5d] loss: %.6f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

print('Finished Training')


# In[ ]:


testset = PosDataset(x_test, y_test)
testset_loader = DataLoader(testset, batch_size=20)


# In[ ]:


x, y = next(iter(testset_loader))
output = net(x)
pprint(list(zip(y, output.data)))


# In[ ]:


output.data


# In[ ]:


correct = 0
total = 0
with torch.no_grad():
    for data in testset_loader:
        x, labels = data
        outputs = net(x)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on testset: %d %%' % (
    100 * correct / total))

