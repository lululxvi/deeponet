"""
@author: jpzxshi
"""
import learner as ln
from data import AntideData, PendData

def antiderivative():
    device = 'gpu' # 'cpu' or 'gpu'
    # data
    T = 1
    s0 = [0]
    sensor_in = 100
    sensor_out = 100
    length_scale = 0.2
    train_num = 1000
    test_num = 10000
    # seq2seq
    cell = 'GRU' # 'RNN', 'LSTM' or 'GRU'
    hidden_size = 5
    # training
    lr = 0.001
    iterations = 50000
    print_every = 1000
    
    data = AntideData(T, s0, sensor_in, sensor_out, length_scale, train_num, test_num)
    net = ln.nn.S2S(data.dim, sensor_in, data.K, sensor_out, hidden_size, cell)
    args = {
        'data': data,
        'net': net,
        'criterion': 'MSE',
        'optimizer': 'adam',
        'lr': lr,
        'iterations': iterations,
        'batch_size': None,
        'print_every': print_every,
        'save': True,
        'callback': None,
        'dtype': 'float',
        'device': device
    }
            
    ln.Brain.Init(**args)
    ln.Brain.Run()
    ln.Brain.Restore()
    ln.Brain.Output()

def pendulum():
    device = 'gpu' # 'cpu' or 'gpu'
    # data
    T = 3
    s0 = [0, 0]
    sensor_in = 100
    sensor_out = 100
    length_scale = 0.2
    train_num = 1000
    test_num = 10000
    # seq2seq
    cell = 'GRU' # 'RNN', 'LSTM' or 'GRU'
    hidden_size = 5
    # training
    lr = 0.001
    iterations = 100000
    print_every = 1000
    
    data = PendData(T, s0, sensor_in, sensor_out, length_scale, train_num, test_num)
    net = ln.nn.S2S(data.dim, sensor_in, data.K, sensor_out, hidden_size, cell)
    args = {
        'data': data,
        'net': net,
        'criterion': 'MSE',
        'optimizer': 'adam',
        'lr': lr,
        'iterations': iterations,
        'batch_size': None,
        'print_every': print_every,
        'save': True,
        'callback': None,
        'dtype': 'float',
        'device': device
    }
            
    ln.Brain.Init(**args)
    ln.Brain.Run()
    ln.Brain.Restore()
    ln.Brain.Output()
        
def main():
    antiderivative()
    #pendulum()
    
if __name__ == '__main__':
    main()