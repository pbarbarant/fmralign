import nibabel as nib
import pandas as pd
import pytest
from nilearn._utils.data_gen import generate_random_img
from nilearn.datasets.tests._testing import dict_to_archive
from numpy.testing import assert_array_equal

from fmralign.fetch_example_data import fetch_ibc_subjects_contrasts


@pytest.fixture
def _mock_metadata_df():
    subs = [
        f"sub-{i:02d}" for i in [1, 2, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15]
    ]
    return pd.DataFrame(
        {
            "subject": sum(([s] * 2 for s in subs), []),
            "condition": ["mock_condition"] * len(subs) * 2,
            "path": [f"path_to_dir/{s}/mock_condition_ap.nii.gz" for s in subs]
            + [f"path_to_dir/{s}/mock_condition_pa.nii.gz" for s in subs],
        }
    )


def test_fetch_ibc_subjects_contrasts(
    tmp_path, request_mocker, _mock_metadata_df
):
    """Test IBC files fetcher"""
    # Mock subject archives
    url_keys = {
        "sub-01": "8z23h",
        "sub-02": "e9kbm",
        "sub-04": "qn5b6",
        "sub-05": "u74a3",
        "sub-06": "83bje",
        "sub-07": "43j69",
        "sub-08": "ua8qx",
        "sub-09": "bxwtv",
        "sub-11": "3dfbv",
        "sub-12": "uat7d",
        "sub-13": "p238h",
        "sub-14": "prdk4",
        "sub-15": "sw72z",
    }
    for sub, key in url_keys.items():
        imgs = {
            f"{sub}/mock_condition_ap.nii.gz": generate_random_img((3, 4, 5))[
                0
            ],
            f"{sub}/mock_condition_pa.nii.gz": generate_random_img((3, 4, 5))[
                0
            ],
        }
        request_mocker.url_mapping[f"https://osf.io/{key}/download"] = (
            dict_to_archive(imgs)
        )

    # Mock metadata CSV
    request_mocker.url_mapping["https://osf.io/pcvje/download"] = (
        dict_to_archive(
            {
                "ibc_3mm_all_subjects_metadata.csv": _mock_metadata_df.to_csv(
                    index=False
                )
            }
        )
    )

    # Mock mask
    mask_img = generate_random_img((3, 4, 5))[1]
    request_mocker.url_mapping["https://osf.io/yvju3/download"] = (
        dict_to_archive({"gm_mask_3mm.nii.gz": mask_img})
    )

    # --- Single subject ---
    files, metadata, mask = fetch_ibc_subjects_contrasts(
        ["sub-01"], data_dir=tmp_path
    )
    assert len(files) == 1
    assert len(files[0]) == len(metadata) == 2
    assert_array_equal(nib.load(mask).get_fdata(), mask_img.get_fdata())

    # --- All subjects ---
    files, metadata, _ = fetch_ibc_subjects_contrasts("all", data_dir=tmp_path)
    assert len(files) == len(url_keys)
    assert len(metadata) == len(url_keys) * 2
