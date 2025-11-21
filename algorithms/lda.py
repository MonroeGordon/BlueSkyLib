import cupy as cp
import numpy as np

class LDA:
    '''
    Linear Discriminant Analysis (LDA) class. Provides algorithms for classification and dimensionality reduction using
    linear discriminant analysis.
    '''

    @staticmethod
    def between_class_scatter(x: np.ndarray | cp.ndarray,
                              y: list,
                              means: dict,
                              device: str = "cpu") -> np.ndarray | cp.ndarray:
        '''
        Compute the between-class scatter matrix.
        :param x: Input feature matrix (number samples, number features).
        :param y: Class labels (number samples).
        :param means: Mean vectors for each class.
        :param device: CPU or GPU device.
        :return: Between-class scatter matrix.
        '''
        if x.ndim != 2:
            raise ValueError("LDA @ between_class_scatter: parameter 'x' must be a matrix (2-dimensional ndarray).")

        if device == "cpu":
            nx = x

            if isinstance(nx, cp.ndarray):
                nx = cp.asnumpy(nx)

            overall_mean = np.mean(nx, axis=0)
            sb = np.zeros((nx.shape[1], nx.shape[1]))

            for c, mean in means.items():
                nc = np.sum(y == c)
                mean = mean.reshape(-1, 1)
                overall_mean = overall_mean.reshape(-1, 1)
                sb += nc * (mean - overall_mean).dot((mean - overall_mean).T)
        else:
            cx = x

            if isinstance(cx, np.ndarray):
                cx = cp.asarray(cx)

            overall_mean = cp.mean(cx, axis=0)
            sb = cp.zeros((cx.shape[1], cx.shape[1]))

            for c, mean in means.items():
                nc = cp.sum(y == c)
                mean = mean.reshape(-1, 1)
                overall_mean = overall_mean.reshape(-1, 1)
                sb += nc * (mean - overall_mean).dot((mean - overall_mean).T)

        return sb

    @staticmethod
    def means(x: np.ndarray | cp.ndarray,
              y: list,
              device: str="cpu") -> dict:
        '''
        Compute the mean vectors for each class.
        :param x: Input feature matrix (number samples, number features)
        :param y: Class labels (number samples)
        :param device: CPU or GPU device.
        :return: Dictionary of mean vectors for each class.
        '''
        if x.ndim != 2:
            raise ValueError("LDA @ means: parameter 'x' must be a matrix (2-dimensional ndarray).")

        if device == "cpu":
            nx = x

            if isinstance(nx, cp.ndarray):
                nx = cp.asnumpy(nx)

            classes = np.unique(y)
            means = {}

            for c in classes:
                means[c] = np.mean(nx[y == c], axis=0)

            return means
        else:
            cx = x

            if isinstance(cx, cp.ndarray):
                cx = cp.asnumpy(cx)

            classes = cp.unique(y)
            means = {}

            for c in classes:
                means[c] = cp.mean(cx[y == c], axis=0)

            return means

    @staticmethod
    def within_class_scatter(x: np.ndarray | cp.ndarray,
                             y: list,
                             means: dict,
                             device: str="cpu") -> np.ndarray | cp.ndarray:
        '''
        Compute the within-class scatter matrix.
        :param x: Input feature matrix (number samples, number features).
        :param y: Class labels (number samples).
        :param means: Mean vectors for each class.
        :param device: CPU or GPU device.
        :return: Within-class scatter matrix.
        '''
        if x.ndim != 2:
            raise ValueError("LDA @ within_class_scatter: parameter 'x' must be a matrix (2-dimensional ndarray).")

        if device == "cpu":
            nx = x

            if isinstance(nx, cp.ndarray):
                nx = cp.asnumpy(nx)

            sw = np.zeros((nx.shape[1], nx.shape[1]))

            for c, mean in means.items():
                class_scatter = np.zeros((nx.shape[1], nx.shape[1]))

                for sample in nx[y == c]:
                    sample = sample.reshape(-1, 1)
                    mean = mean.reshape(-1, 1)
                    class_scatter += (sample - mean).dot((sample - mean).T)

                sw += class_scatter
        else:
            cx = x

            if isinstance(cx, np.ndarray):
                cx = cp.asarray(cx)

            sw = cp.zeros((cx.shape[1], cx.shape[1]))

            for c, mean in means.items():
                class_scatter = cp.zeros((cx.shape[1], cx.shape[1]))

                for sample in cx[y == c]:
                    sample = sample.reshape(-1, 1)
                    mean = mean.reshape(-1, 1)
                    class_scatter += (sample - mean).dot((sample - mean).T)

                sw += class_scatter

        return sw