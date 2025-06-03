"""Module for sparse template alignment."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from fmralign._utils import get_modality_features
from fmralign.alignment_methods import OptimalTransportAlignment


def _align_images_to_template(
    subjects_data,
    template,
    subjects_estimators,
):
    """Align the subjects data to the template using sparse alignment.

    Parameters
    ----------
    subjects_data : List of torch.Tensor of shape (n_samples, n_features)
        List of subjects data.
    template : torch.Tensor of shape (n_samples, n_features)
        Template data.
    subjects_estimators : List of alignment_methods.Alignment
        List of sparse alignment estimators.

    Returns
    -------
    Tuple of List of torch.Tensor of shape (n_samples, n_features)
        and List of alignment_methods.Alignment
        Updated subjects data and alignment estimators.
    """
    n_subjects = len(subjects_data)
    for i in range(n_subjects):
        sparse_estimator = subjects_estimators[i]
        sparse_estimator.fit(subjects_data[i], template)
        subjects_data[i] = sparse_estimator.transform(subjects_data[i])
        # Update the estimator in the list
        subjects_estimators[i] = sparse_estimator
    return subjects_data, subjects_estimators


def alignment_to_template(
    imgs,
    template_data,
    masker,
    parcellation_img=None,
    modality="response",
    verbose=False,
    **kwargs,
):
    alignment_estimators = []
    for img in imgs:
        img_data = get_modality_features(
            [img], parcellation_img, masker, modality=modality
        )[0]
        img_data = masker.transform(img)
        estimator = OptimalTransportAlignment(verbose=verbose, **kwargs)
        estimator.fit(img_data, template_data)
        alignment_estimators.append(estimator)

    return alignment_estimators


def _fit_online_template(
    imgs,
    masker,
    parcellation_img=None,
    modality="response",
    n_iter=100,
    verbose=False,
    **kwargs,
):
    """Fit a the template to the subjects data using sparse alignment.

    Parameters
    ----------
    subjects_data : list of torch.Tensor of shape (n_samples, n_features)
        List of subjects data.
    sparsity_mask : torch sparse COO tensor
        Sparsity mask for the alignment matrix.
    alignment_method : str, optional
        Sparse alignment method, by default "sparse_uot"
    n_iter : int, optional
        Number of template updates, by default 2
    verbose : bool, optional
        Verbosity level, by default False

    Returns
    -------
    Tuple[torch.Tensor, List[alignment_methods.Alignment]]
        Template data and list of alignment estimators
        from the subjects data to the template.

    Raises
    ------
    ValueError
        Unknown alignment method.
    """
    # Initialize the template as the first image
    template_data = masker.transform(imgs[np.random.randint(len(imgs))])

    # Perform stochastic gradient descent to find the template
    estimator = OptimalTransportAlignment(**kwargs)
    n_iter_ = max(n_iter, len(imgs))
    for i in range(n_iter_):
        if verbose:
            print(f"Iteration {i + 1}/{n_iter_}")
        # Get a random image from the subjects
        current_img = imgs[np.random.randint(len(imgs))]
        img_data = get_modality_features(
            [current_img], parcellation_img, masker, modality=modality
        )[0]
        img_data = masker.transform(current_img)
        estimator.fit(img_data, template_data)
        alpha = 1 / (i + 2)
        template_data = (
            1 - alpha
        ) * template_data + alpha * estimator.transform(img_data)
    return template_data


class OnlineTemplateAlignment(BaseEstimator, TransformerMixin):
    """
    Decompose the source images into regions and summarize subjects information
    in a template, then use pairwise alignment to predict
    new contrast for target subject.
    """

    def __init__(
        self,
        alignment_method="sparse_uot",
        n_pieces=1,
        clustering="kmeans",
        n_iter=2,
        save_template=None,
        masker=None,
        modality="response",
        device="cpu",
        n_jobs=1,
        verbose=0,
        **kwargs,
    ):
        """
        Parameters
        ----------
        alignment_method: string
            Algorithm used to perform alignment between source images
            and template, currently only 'sparse_uot' is supported.
        n_pieces: int, optional (default = 1)
            Number of regions in which the data is parcellated for alignment.
            If 1 the alignment is done on full scale data.
            If > 1, the voxels are clustered and alignment is performed on each
            cluster applied to X and Y.
        clustering : string or 3D Niimg optional (default : kmeans)
            'kmeans', 'ward', 'rena', 'hierarchical_kmeans' method used for
            clustering of voxels based on functional signal,
            passed to nilearn.regions.parcellations
            If 3D Niimg, image used as predefined clustering,
            n_pieces is then ignored.
        n_iter: int
           number of iteration in the alternate minimization. Each img is
           aligned n_iter times to the evolving template. If n_iter = 0,
           the template is simply the mean of the input images.
        save_template: None or string(optional)
            If not None, path to which the template will be saved.
        masker : None or :class:`~nilearn.maskers.NiftiMasker` or \
                :class:`~nilearn.maskers.MultiNiftiMasker`, or \
                :class:`~nilearn.maskers.SurfaceMasker` , optional
            A mask to be used on the data. If provided, the mask
            will be used to extract the data. If None, a mask will
            be computed automatically with default parameters.
        modality : str, optional (default='response')
            Specifies the alignment modality to be used:
            * 'response': Aligns by directly comparing corresponding similar 
            time points in the source and target images.
            * 'connectivity': Aligns based on voxel-wise connectivity features 
            within each parcel, comparing how each voxel relates to others in 
            the same region.
            * 'hybrid': Combines both time series and connectivity information 
            to perform the alignment.
        device: string, optional (default = 'cpu')
            Device on which the computation will be done. If 'cuda', the
            computation will be done on the GPU if available.
        n_jobs: integer, optional (default = 1)
            The number of CPUs to use to do the computation. -1 means
            'all CPUs', -2 'all CPUs but one', and so on.
        verbose: integer, optional (default = 0)
            Indicate the level of verbosity. By default, nothing is printed.

        """
        self.template = None
        self.template_history = None
        self.alignment_method = alignment_method
        self.n_pieces = n_pieces
        self.clustering = clustering
        self.n_iter = n_iter
        self.save_template = save_template
        self.masker = masker
        self.modality = modality
        self.device = device
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.kwargs = kwargs

    def fit(self, imgs):
        """
        Learn a template from source images, using alignment.

        Parameters
        ----------
        imgs: List of `str` paths.
            Source subjects data. Each element of the parent list is one subject
            data, and all must have the same length (n_samples).

        Returns
        -------
        self

        Attributes
        ----------
        self.template: 4D Niimg object
            Length : n_samples

        """

        # # Add new features based on the modality
        # imgs_ = get_modality_features(
        #     imgs, self.clustering, self.masker, self.modality
        # )

        template_data = _fit_online_template(
            imgs=imgs,
            masker=self.masker,
            parcellation_img=self.clustering,
            modality=self.modality,
            n_iter=self.n_iter,
            verbose=max(0, self.verbose - 1),
            **self.kwargs,
        )

        self.template = self.masker.inverse_transform(template_data)
        self.fit_ = alignment_to_template(
            imgs=imgs,
            template_data=template_data,
            masker=self.masker,
            modality=self.modality,
            device=self.device,
            n_jobs=self.n_jobs,
            verbose=max(0, self.verbose - 1),
        )

        if self.save_template is not None:
            self.template.to_filename(self.save_template)

    def transform(self, img, subject_index=None):
        """
        Transform a (new) subject image into the template space.

        Parameters
        ----------
        img: 4D Niimg-like object
            Subject image.
        subject_index: int, optional (default = None)
            Index of the subject to be transformed. It should
            correspond to the index of the subject in the list of
            subjects used to fit the template. If None, a new
            `PairwiseAlignment` object is fitted between the new
            subject and the template before transforming.


        Returns
        -------
        predicted_imgs: 4D Niimg object
            Transformed data.

        """
        if not hasattr(self, "fit_"):
            raise ValueError(
                "This instance has not been fitted yet. "
                "Please call 'fit' before 'transform'."
            )

        if subject_index is None:
            raise NotImplementedError(
                "Transforming a new subject without specifying "
                "the subject index is not implemented yet. "
                "Please specify the subject index."
            )
        else:
            alignment_estimator = self.fit_[subject_index]
            img_data = self.masker.transform(img)
            X_transformed = alignment_estimator.transform(img_data)
            return self.masker.inverse_transform(X_transformed)

    # Make inherited function harmless
    def fit_transform(self):
        """Parent method not applicable here. Will raise AttributeError if called."""
        raise AttributeError(
            "type object 'PairwiseAlignment' has no attribute 'fit_transform'"
        )

    def get_parcellation(self):
        """Get the parcellation masker used for alignment.

        Returns
        -------
        labels: `list` of `int`
            Labels of the parcellation masker.
        parcellation_img: Niimg-like object
            Parcellation image.
        """
        if hasattr(self, "parcel_masker"):
            check_is_fitted(self)
            labels = self.parcel_masker.get_labels()
            parcellation_img = self.parcel_masker.get_parcellation_img()
            return labels, parcellation_img
        else:
            raise AttributeError(
                (
                    "Parcellation has not been computed yet,"
                    "please fit the alignment estimator first."
                )
            )
