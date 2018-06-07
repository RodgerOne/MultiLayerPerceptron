import numpy as np
import pickle as pkl
from MultiLayerPerceptron import SiecNeuronowa

########  OLD FUNCTIONS _###############

def load_data_org():
    PICKLE_FILE_PATH = 'train.pkl'
    with open(PICKLE_FILE_PATH, 'rb') as f:
        return pkl.load(f)


def save_data(data):
    PICKLE_FILE_PATH = 'trained_data_model.pkl'
    with open(PICKLE_FILE_PATH, 'wb') as f:
        pkl.dump(data, f)
    pass


def scale_image(img):
    tmp = img.reshape((56, 56))
    x = 28
    new = np.zeros((x,x))
    for i in range(x):
        for j in range(x):
            c = np.unique(tmp[2*i:2*(i+1), 2*j:2*(j+1)], return_counts=True)
            new[i][j] =  c[0][np.argmax(c[1])]
    return new


def my_features(x):  # cos co mozna nazwaz ekstrakcja cech, a raczej reskalowaniem obrazu
    sum_size = 7
    result = np.ones([x.shape[0], int(x.shape[1] / sum_size)])
    for j in range(sum_size, x.shape[1] - sum_size, sum_size):
        sum = np.sum(x[:, j - (sum_size - 1):j + 1], 1)
        result[:, int(j / sum_size)] = sum  >= 1
    return result


def max_pooling(x):  # cos co mozna nazwaz ekstrakcja cech, a raczej reskalowaniem obrazu
    sum_size = 7
    result = np.ones([x.shape[0], int(x.shape[1] / sum_size)])
    for j in range(sum_size, 56 - sum_size, sum_size):
        for i in range(sum_size, 56 - sum_size, sum_size):
            result[:, int((j / sum_size)*56 + (i/sum_size))] = np.max(x[:, j - (sum_size -1): j+1], axis=1)
    return result


def hog(img):
    image = img.reshape(56, 56) #7,7,15 dobre dla 56,56, lipa
    nwin_x = 7
    nwin_y = 7
    B = 11
    L, C = np.shape(image)
    H = np.zeros(shape=(nwin_x*nwin_y*B,1))
    step_x = np.floor(C/(nwin_x+1))
    step_y = np.floor(L/(nwin_y+1))
    cont = 0
    h_xy = np.array([1, 0, -1]) # wspólny dla obu
    grad_xr = np.convolve(image.flatten(), h_xy, mode='same').reshape(56, 56)
    grad_yu = np.convolve(image.T.flatten(), h_xy, mode='same').reshape(56, 56).T
    angles = np.arctan2(grad_yu,grad_xr)
    magnit = np.sqrt((grad_yu**2 +grad_xr**2))
    for n in range(nwin_y):
        for m in range(nwin_x):
            cont += 1
            angles2 = angles[int(n*step_y):int((n+2)*step_y),int(m*step_x):int((m+2)*step_x)]
            magnit2 = magnit[int(n*step_y):int((n+2)*step_y),int(m*step_x):int((m+2)*step_x)]
            v_angles = angles2.ravel()
            v_magnit = magnit2.ravel()
            bin = 0
            H2 = np.zeros(shape=(B,1))
            for ang_lim in np.arange(start=-np.pi+2*np.pi/B,stop=np.pi+2*np.pi/B,step=2*np.pi/B):
                check = v_angles < ang_lim
                v_angles = (v_angles*(~check)) + (check) * 100
                H2[bin] += np.sum(v_magnit * check)
                bin += 1
            H2 /= (np.linalg.norm(H2)+0.01)
            H[(cont-1)*B:cont*B]=H2
    return H.flatten()


def hogs(x):
    return np.apply_along_axis(hog, 1, x)


def squares(X):
    side = 56
    step = 7
    coef = int(side / step)
    result = np.zeros((X.shape[0], int(X.shape[1]/(step ** 2))))
    for i in range(X.shape[0]):
        img = X[i].reshape(56, 56)
        for x in range(coef):
            for y in range(coef):
                result[i][x*coef + y] = int(np.sum(img[step*x : step*(x+1), step*y : step*(y+1)])/step)
    return result

###########################################################

from predict import predict
X_train, y_train = load_data_org()
X_train = X_train
y_train = y_train

X_val, y_val = load_data_org()
X_val = X_val[:2500]
y_val = y_val[:2500]

nn = SiecNeuronowa(n_output=36, n_features=X_train.shape[1], n_hidden=240, lambda_2=0.1, lambda_1=0.0, epochs=1000, eta=0.001, alpha=0.001, decrease_const=0.00001, shuffle=True, minibatches=43, random_state=1)
nn.train_net(X_train, y_train, print_progress=True)

#nn.print_model_params()

a = res1, res2 = np.unique(predict(X_val) == y_val, return_counts=True)
print('Skuteczność: ' + str(res2[1]/(res2[0] + res2[1]) * 100) + ' % ')

