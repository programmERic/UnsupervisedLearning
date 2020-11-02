
import numpy as np
SEED=np.random.seed(12)

import matplotlib.pyplot as plt
from factor_analyzer import FactorAnalyzer
import time
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA, FastICA, FactorAnalysis
from sklearn.metrics import accuracy_score, f1_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.random_projection import SparseRandomProjection
from scipy.spatial.distance import cdist


def heart_failure_data(filename='heart_failure_clinical_records_dataset.csv'):
    '''
    Heart Disease
    '''
    heart_failure = np.genfromtxt(filename, delimiter=',')

    heart_failure = heart_failure[1:, :]

    heart_X = heart_failure[:, :-1]
    heart_y = np.reshape(heart_failure[:, -1], (-1, 1))
    
    return heart_X, heart_y

def heart_data(test_size=0.4):
    X, y = heart_failure_data()
    #X_test, X_train, y_test, y_train = train_test_split(X, y, test_size=test_size, random_state=SEED)

    X_scaled = StandardScaler().fit_transform(X)

    return X_scaled, y

def digits_data(test_size=0.4):
    
    nums = load_digits()
    X, y = nums.data, nums.target

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
        
    #X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=SEED)

    #return X_test, X_train, y_test, y_train

    return X_scaled, y

def kmeans_experiment():

    X_digits, y_digits = digits_data()
    X_heart, y_heart = heart_data()
    X = X_digits
    kmeans = KMeans(n_clusters=32, random_state=SEED)
    y_pred = kmeans.fit_predict(X)
    

    ax = plt.gca()
    ax.axis('equal')
    ax.scatter(X[:, 0], X[:, 1], c=y_pred, s=40, cmap='viridis', zorder=2)

    centers = kmeans.cluster_centers_
    radii = [cdist(X[y_pred == i], [center]).max()
             for i, center in enumerate(centers)]
    for c, r in zip(centers, radii):
        ax.add_patch(plt.Circle(c, r, fc='#CCCCCC', lw=3, alpha=0.5, zorder=1))
    plt.title("kMeans clusters (Digits Data)")
    plt.savefig("kmeans_2d_cluster_digits.png")

    plt.show()

def determine_kmeans_inertia():

    X_scaled, y = heart_data()
    max_clusters = X_scaled.shape[1]
    
    inertia = np.zeros(shape=max_clusters-1)

    for num_cluster in range(1, max_clusters):
        kmeans = KMeans(n_clusters=num_cluster, random_state=SEED).fit(X_scaled)
        inertia[num_cluster-1] = kmeans.inertia_

    
    plt.figure()
    plt.plot(inertia)
    plt.title("kMeans Inertia for various k (Heart Disease Data)")
    plt.xlabel("Number of clusters, k")
    plt.ylabel("Squared Distance to Cluster")
    plt.savefig("inertia_kmeans_heart.png")


def expectation_max_experiment():
    
    X_scaled, y = digits_data()
    max_clusters = X_scaled.shape[1]
    
    n_components = np.arange(1, max_clusters)
    models = [GaussianMixture(n, covariance_type='full', random_state=SEED).fit(X_scaled) for n in n_components]

    fig = plt.figure()
    plt.plot(n_components, [m.bic(X_scaled) for m in models], label='BIC')
    plt.plot(n_components, [m.aic(X_scaled) for m in models], label='AIC')
    plt.title("EM info. coeff. for various # of components (Digits Data)")
    plt.legend(loc='best')
    plt.xlabel('n_components')
    plt.savefig("EM_components_v_IC_digits_data.png")
    plt.show()

def srp_experiment():

    X_digits, y = digits_data()
    X_digits_test, X_digits_train, y_test, y_train = train_test_split(X_digits, y, test_size=0.4, random_state=SEED)

    test = range(2,64)

    if False: # skip this for right now
        for n in test:

            srp = SparseRandomProjection(n_components=n)
            srp_digits = srp.fit_transform(X_digits_train)

            colors = ['red', 'blue', 'green', 'coral', 'khaki', 'yellow', 'turquoise', 'pink', 'moccasin', 'olive']
            
            plt.figure()

            for i in range(10):
                plt.scatter(srp_digits[y_train==i, 0], srp_digits[y_train==i, 1], color=colors[i], alpha=0.5,label=i)

            plt.title("Sparse Random Projections, n_components={} (Digits data)".format(n))
            plt.ylabel('Subcomponent 2')
            plt.xlabel('Subcomponent 1')
            plt.legend()
            plt.savefig("folder/SRP_digits_{}.png".format(n))

    X_heart, y_heart = heart_data()
    X_heart_test, X_heart_train, y_heart_test, y_heart_train = train_test_split(X_heart, y_heart, test_size=0.4, random_state=SEED)

    for n in range(2, 12):

        srp2 = SparseRandomProjection(n_components=n, random_state=SEED)
        srp_heart = srp2.fit_transform(X_heart_train)
        colors = ['red', 'blue']
        
        plt.figure()
        for i in range(2):
            plt.scatter(srp_heart[y_heart_train[:,0]==i, 0], srp_heart[y_heart_train[:,0]==i, 1], color=colors[i], alpha=0.5,label=i)

        plt.title("SRP, n_components={} (Heart data)".format(n))
        plt.ylabel('Subcomponent 2')
        plt.xlabel('Subcomponent 1')
        plt.legend()
        plt.savefig("folder/heart_SRP_{}.png".format(n))

def ica_experiment():

    X_digits, y = digits_data()
    X_digits_test, X_digits_train, y_test, y_train = train_test_split(X_digits, y, test_size=0.4, random_state=SEED)

    ica = FastICA(n_components=3, random_state=SEED)
    ica_digits = ica.fit_transform(X_digits_train)

    colors = ['red', 'blue', 'green', 'coral', 'khaki', 'yellow', 'turquoise', 'pink', 'moccasin', 'olive']
    
    plt.figure()

    for i in range(10):
        plt.scatter(ica_digits[y_train==i, 0], ica_digits[y_train==i, 1], color=colors[i], alpha=0.5,label=i)

    plt.title("ICA (Digits data)")
    plt.ylabel('Subcomponent 2')
    plt.xlabel('Subcomponent 1')
    plt.legend()
    plt.savefig("ICA_digits.png")


    X_heart, y_heart = heart_data()
    X_heart_test, X_heart_train, y_heart_test, y_heart_train = train_test_split(X_heart, y_heart, test_size=0.4, random_state=SEED)

    ica2 = FastICA(n_components=6, random_state=SEED)
    ica_heart = ica2.fit_transform(X_heart_train)
    colors = ['red', 'blue']
    
    plt.figure()
    for i in range(2):
        plt.scatter(ica_heart[y_heart_train[:,0]==i, 0], ica_heart[y_heart_train[:,0]==i, 1], color=colors[i], alpha=0.5,label=i)

    plt.title("ICA (Heart data)")
    plt.ylabel('Subcomponent 2')
    plt.xlabel('Subcomponent 1')
    plt.legend()
    plt.savefig("ICA_heart.png")
    plt.show()
    
def cluster_experiment():

    X_digits, y_digits = digits_data()

    kmeans = KMeans(n_clusters=32, random_state=SEED)
    y_pred = kmeans.fit_predict(X_digits)
 
    centers = kmeans.cluster_centers_
    
    ax = plt.gca(centers)
    ax.axis('equal')
    ax.scatter(X[:, 0], X[:, 1], c=y_pred, s=40, cmap='viridis', zorder=2)

    centers = kmeans.cluster_centers_
    radii = [cdist(X[y_pred == i], [center]).max()
             for i, center in enumerate(centers)]
    for c, r in zip(centers, radii):
        ax.add_patch(plt.Circle(c, r, fc='#CCCCCC', lw=3, alpha=0.5, zorder=1))
    plt.title("kMeans clusters (Digits Data)")
    plt.savefig("kmeans_2d_cluster_digits.png")


def dimensionality_experiment():

    X, y = heart_data()
    #covariance = [0.5, 0.75, 0.8, 0.875, 0.95]

    num_components = [3, 6, 9, 12]

    labels = ["Not-Linked", "Linked"]
    targets = [0, 1]
    colors = ['r', 'g']

    for num in num_components:

        pca = PCA(n_components=num, random_state=SEED)
        principalComponents = pca.fit_transform(X)

        print(pca.n_components_)

        fig = plt.figure(figsize = (8,8))
        ax = fig.add_subplot(1,1,1) 
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_title('{num} component PCA'.format(num=num), fontsize=20)
        
        for target, color in zip(targets,colors):
            label = (y[:] == target).reshape(-1,)
            ax.scatter(principalComponents[label, 0], principalComponents[label, 1], c=color, s=25)
        ax.legend(labels)
        ax.grid()

        plt.savefig("pca_{num}_heart.png".format(num=num))


def PCA_diminsionality_variance_graph():

    X_heart, y = heart_data()
    X_digits, y = digits_data()

    pca_heart = PCA().fit(X_heart)
    pca_digits = PCA().fit(X_digits)

    plt.plot(np.cumsum(pca_heart.explained_variance_ratio_), label="Heart Data")
    plt.plot(np.cumsum(pca_digits.explained_variance_ratio_), label="Digits Data")
    plt.legend(loc='best')
    plt.grid()
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.savefig("pca_variance_v_components.png")
  
def PCA_2_colormap():

    X_heart, y_heart = heart_data()
    pca_heart = PCA(n_components=2).fit_transform(X_heart)

    fig = plt.figure()
    
    plt.scatter(pca_heart[:, 0], pca_heart[:, 1],
            c=y_heart, edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('RdBu', 2))
    plt.title("PCA clusters with 2 components (Heart Disease Data)")
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.colorbar()
    plt.grid()
    plt.savefig("pca_2_components_heart.png")

    X_digits, y_digits = digits_data()
    pca_digits = PCA(n_components=2).fit_transform(X_digits)
    
    fig = plt.figure()
    plt.scatter(pca_digits[:, 0], pca_digits[:, 1],
            c=y_digits, edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('gist_rainbow', 10))
    plt.title("PCA clusters with 2 components (Digits Data)")
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.colorbar()
    plt.grid()
    plt.savefig("pca_2_components_digits.png")
    
    

def random_projection_experiment():
    
    X_heart, y = digits_data()
    transformer = GaussianRandomProjection(n_components=24, random_state=SEED)
    X_new = transformer.fit_transform(X_heart)

    print(X_heart.shape)
    print(X_new.shape)


def plot_digits_reconstructed():

    nums = load_digits()
    X_digits, y = nums.data, nums.target

    fig, axes = plt.subplots(8, 10, figsize=(10, 8),
                             subplot_kw={'xticks':[], 'yticks':[]},
                             gridspec_kw=dict(hspace=0.1, wspace=0.1))

    var = [0.975, 0.9, 0.8, 0.7, 0.5, 0.33, 0.166]

    for i, ax in enumerate(axes.flat):

        index = i // 10 - 1

        if index == -1:
            ax.imshow(X_digits[i].reshape(8, 8), cmap='binary', interpolation='nearest', clim=(0, 16))
            ax.set(ylabel='Original')
        else:
            curr_pca = PCA(var[index]).fit(X_digits)
            vals = curr_pca.inverse_transform(curr_pca.transform(X_digits[i].reshape(1,-1)))
            ax.imshow(vals.reshape(8, 8), cmap='binary', interpolation='nearest', clim=(0, 16))

            ax.set(ylabel='{}'.format(var[index]))
        ax.label_outer()

    fig.suptitle("PCA Reconstruction")
    plt.savefig("PCA_reconstruction_digits.png")        
    plt.show()

def factor_analysis():

    X_heart, y = heart_data()
    
    factor = FactorAnalyzer()
    factor.fit(X_heart)

    eigen, v = factor.get_eigenvalues()
    plt.plot(range(1,X_heart.shape[1]+1),eigen)
    plt.hlines(1.0, 0, X_heart.shape[1]+1, color="orange", linestyles='dashed')

    plt.title("Factor Analysis (Heart Disease Data)")
    plt.xlabel('Factor')
    plt.ylabel('Eigen Value')
    plt.grid()
    plt.savefig("factor_analysis_heart.png")

def neuralnet_after_reduction():

    X_digits, y = digits_data()
    X_digits_test, X_digits_train, y_test, y_train = train_test_split(X_digits, y, test_size=0.4, random_state=SEED)

    max_iter = 3000
    
    print("Neural Network Timing Experiment using Digits data.")
    start = time.time()
    nn = MLPClassifier(random_state=SEED, hidden_layer_sizes=(3,3), max_iter=max_iter)
    nn.fit(X_digits_train, y_train)
    finish = time.time()
    y_pred = nn.predict(X_digits_test)
    print("Neural network no dimension reduction")
    print("    Fit time:     ", finish-start, "sec.")
    print("    Test Accurracy", accuracy_score(y_test, y_pred))
    print()
    
    pca = PCA(n_components=32, random_state=SEED)
    pca_digits = pca.fit_transform(X_digits_train)
    start = time.time()
    nn = MLPClassifier(random_state=SEED, hidden_layer_sizes=(3,3), max_iter=max_iter)
    nn.fit(pca_digits, y_train)
    finish = time.time()
    pca_transformed = pca.transform(X_digits_test)
    y_pred = nn.predict(pca_transformed)
    print("Neural network with PCA reduced data")
    print("    Fit time:     ", finish-start, "sec.")
    print("    Test Accurracy", accuracy_score(y_test, y_pred))
    print()

    ica = FastICA(n_components=31, random_state=SEED)
    ica_digits = ica.fit_transform(X_digits_train)
    start = time.time()
    nn = MLPClassifier(random_state=SEED, hidden_layer_sizes=(3,3), max_iter=max_iter)
    nn.fit(ica_digits, y_train)
    finish = time.time()
    ica_transformed = ica.transform(X_digits_test)
    y_pred = nn.predict(ica_transformed)
    print("Neural network with ICA reduced data")
    print("    Fit time:     ", finish-start, "sec.")
    print("    Test Accurracy", accuracy_score(y_test, y_pred))
    print()
    keep = accuracy_score(y_test, y_pred)
    
    srp = SparseRandomProjection(n_components=16, random_state=SEED)
    srp_digits = srp.fit_transform(X_digits_train)
    start = time.time()
    nn = MLPClassifier(random_state=SEED, hidden_layer_sizes=(3,3), max_iter=max_iter)
    nn.fit(srp_digits, y_train)
    finish = time.time()
    srp_transformed = srp.transform(X_digits_test)
    y_pred = nn.predict(srp_transformed)
    print("Neural network with random projection reduced data")
    print("    Fit time:     ", finish-start, "sec.")
    print("    Test Accurracy", accuracy_score(y_test, y_pred))
    print()

    
    fa = FactorAnalysis(n_components=6, random_state=SEED)
    fa_digits = fa.fit_transform(X_digits_train)
    start = time.time()
    nn = MLPClassifier(random_state=SEED, hidden_layer_sizes=(3,3), max_iter=max_iter)
    nn.fit(fa_digits, y_train)
    finish = time.time()
    fa_transformed = fa.transform(X_digits_test)
    y_pred = nn.predict(fa_transformed)
    print("Neural network with factor analysis reduced data")
    print("    Fit time:     ", finish-start, "sec.")
    print("    Test Accurracy", accuracy_score(y_test, y_pred))
    print()

def neuralnet_after_reduction_and_clustering():

    X_digits, y = digits_data()
    X_digits_test, X_digits_train, y_test, y_train = train_test_split(X_digits, y, test_size=0.4, random_state=SEED)

    max_iter = 3000
    
    print("Neural Network Timing Experiment using Digits data.")
    start = time.time()
    nn = MLPClassifier(random_state=SEED, hidden_layer_sizes=(3,3), max_iter=max_iter)
    nn.fit(X_digits_train, y_train)
    finish = time.time()
    y_pred = nn.predict(X_digits_test)
    print("Neural network no dimension reduction")
    print("    Fit time:     ", finish-start, "sec.")
    print("    Test Accurracy", accuracy_score(y_test, y_pred))
    print()
    
    pca = PCA(n_components=32, random_state=SEED)
    pca_digits = pca.fit_transform(X_digits_train)
    kmeans = KMeans(n_clusters=32, random_state=SEED)
    kmeans.fit(pca_digits)
    k_data = kmeans.transform(pca_digits)
    start = time.time()
    nn = MLPClassifier(random_state=SEED, hidden_layer_sizes=(3,3), max_iter=max_iter)
    nn.fit(k_data, y_train)
    finish = time.time()
    kmeans_transformed = kmeans.transform(pca.transform(X_digits_test))
    y_pred = nn.predict(kmeans_transformed)
    print("Neural network with PCA reduced data and kMeans Clustering")
    print("    Fit time:     ", finish-start, "sec.")
    print("    Test Accurracy", accuracy_score(y_test, y_pred))
    print()

    ica = FastICA(n_components=31, random_state=SEED)
    ica_digits = ica.fit_transform(X_digits_train)
    kmeans = KMeans(n_clusters=31, random_state=SEED)
    kmeans.fit(ica_digits)
    k_data = kmeans.transform(ica_digits)
    start = time.time()
    nn = MLPClassifier(random_state=SEED, hidden_layer_sizes=(3,3), max_iter=max_iter)
    nn.fit(k_data, y_train)
    finish = time.time()
    kmeans_transformed = kmeans.transform(ica.transform(X_digits_test))
    y_pred = nn.predict(kmeans_transformed)
    print("Neural network with ICA reduced data and kMeans Clustering")
    print("    Fit time:     ", finish-start, "sec.")
    print("    Test Accurracy", accuracy_score(y_test, y_pred))
    print()
    keep = accuracy_score(y_test, y_pred)
    
    srp = SparseRandomProjection(n_components=16, random_state=SEED)
    srp_digits = srp.fit_transform(X_digits_train)
    kmeans = KMeans(n_clusters=16, random_state=SEED)
    kmeans.fit(srp_digits)
    k_data = kmeans.transform(srp_digits)
    start = time.time()
    nn = MLPClassifier(random_state=SEED, hidden_layer_sizes=(3,3), max_iter=max_iter)
    nn.fit(k_data, y_train)
    finish = time.time()
    kmeans_transformed = kmeans.transform(srp.fit_transform(X_digits_test))
    y_pred = nn.predict(kmeans_transformed)
    print("Neural network with random projection reduced data and kMeans Clustering")
    print("    Fit time:     ", finish-start, "sec.")
    print("    Test Accurracy", accuracy_score(y_test, y_pred))
    print()

    
    fa = FactorAnalysis(n_components=6, random_state=SEED)
    fa_digits = fa.fit_transform(X_digits_train)
    kmeans = KMeans(n_clusters=6, random_state=SEED)
    kmeans.fit(fa_digits)
    k_data = kmeans.transform(fa_digits)
    start = time.time()
    nn = MLPClassifier(random_state=SEED, hidden_layer_sizes=(3,3), max_iter=max_iter)
    nn.fit(k_data, y_train)
    finish = time.time()
    kmeans_transformed = kmeans.transform(fa.fit_transform(X_digits_test))
    y_pred = nn.predict(kmeans_transformed)
    print("Neural network with factor analysis reduced data and kMeans Clustering")
    print("    Fit time:     ", finish-start, "sec.")
    print("    Test Accurracy", accuracy_score(y_test, y_pred))
    print()

if __name__ == "__main__":
    factor_analysis()
    neuralnet_after_reduction()
    neuralnet_after_reduction_and_clustering()

    
