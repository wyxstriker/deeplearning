import model
import data_load
import torch.utils.data.dataloader as dl
import torch

if __name__=='__main__':
    model = model.my_AlexNet()
    path = './caltech'
    batchSize = 64
    testSet = dl.DataLoader(data_load.CaltechSet(path, 'test'), batch_size=batchSize, shuffle=False)
    model.load_state_dict(torch.load('weights.tar'))
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testSet:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the all test images: %.3f %%' % (
            100.0 * correct / total))