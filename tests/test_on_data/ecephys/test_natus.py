"""
Tests of NatusRecordingInterface on local Natus/XLTEK sample sessions.

The sample data is not public, so the tests are skipped unless these environment
variables are set (the same as for neo's test_natusrawio.py):

* ``NATUS_TEST_ROOT``: folder holding ``Natus_deident1`` and ``Natus_deident2``.
* ``NATUS_DEV_ROOT`` (optional): folder holding the de-identified development sessions;
  used for the session whose sample rate changes between files.
"""

import os
from pathlib import Path

import numpy as np
import pytest

SAMPLE_ROOT = os.environ.get("NATUS_TEST_ROOT")
DEV_ROOT = os.environ.get("NATUS_DEV_ROOT")

PRUNED = "Natus_deident2/Xxxxxxxx~ Xxxx_42a53fa4-f077-4d5b-bcb3-6446aecae1c1"  # 24 segments, 256 Hz, 46 ch
DEPTH = "Natus_deident1/Xxxx~ Xxxx_85ba4df1-98ef-49fc-9e67-ae3d1f5ad0b3"  # depth leads, first channel LA1
RATE_CHANGE_SESSION = "LastNameA~ FirstNameA_c0fe2d81-a1d2-4da9-818d-481ecbe0a4b1"

pytestmark = pytest.mark.skipif(not SAMPLE_ROOT, reason="NATUS_TEST_ROOT not set")


def session_dir(relative, root=None):
    d = Path(root or SAMPLE_ROOT) / relative
    if not d.is_dir():
        pytest.skip(f"session folder missing: {d}")
    return d


def caller_metadata(interface):
    """The metadata a caller completes: the subject is not the interface's to fill in."""
    metadata = interface.get_metadata()
    metadata["NWBFile"]["session_description"] = "Natus test session."
    metadata["NWBFile"]["experimenter"] = ["Tester, Test"]
    metadata["NWBFile"]["institution"] = "Test"
    metadata["Subject"] = dict(subject_id="MSEL00000", species="Homo sapiens", sex="U", age="P30Y")
    return metadata


def test_lead_name():
    from neuroconv.datainterfaces.ecephys.natus.natusdatainterface import _lead_name

    assert _lead_name("LA1") == "LA"
    assert _lead_name("LA12") == "LA"
    assert _lead_name("Fp1") == "Fp"
    assert _lead_name("EKG") == "EKG"
    assert _lead_name("DC-01") == "DC"
    assert _lead_name("12") == "12"


def test_registered():
    from neuroconv.datainterfaces import NatusRecordingInterface, interfaces_by_category

    assert interfaces_by_category["ecephys"]["Natus"] is NatusRecordingInterface


def test_metadata_and_groups():
    from neuroconv.datainterfaces import NatusRecordingInterface

    d = session_dir(DEPTH)
    interface = NatusRecordingInterface(folder_path=d)
    metadata = interface.get_metadata()

    from neo.rawio import NatusRawIO

    r = NatusRawIO(dirname=d)
    r.parse_header()
    t0 = r.raw_annotations["blocks"][0]["rec_datetime"]
    assert metadata["NWBFile"]["session_start_time"] == t0
    assert metadata["NWBFile"]["session_start_time"].tzinfo is not None
    assert "Subject" not in metadata or "subject_id" not in metadata["Subject"]

    recording = interface.recording_extractor
    names = list(recording.get_property("channel_name"))
    groups = list(recording.get_property("group_name"))
    assert names[0] == "LA1" and groups[0] == "LA"
    assert set(metadata["Ecephys"]["ElectrodeGroups"]) == set(groups)
    assert metadata["Ecephys"]["ElectrodeGroups"]["LA"]["device_metadata_key"] in metadata["Devices"]

    # no patient name from the folder anywhere in the metadata
    folder_name_part = d.name.split("_")[0]
    assert folder_name_part not in repr(metadata)

    # the old list-based format too
    old = interface.get_metadata(use_new_metadata_format=False)
    assert {g["name"] for g in old["Ecephys"]["ElectrodeGroup"]} == set(groups)


@pytest.mark.parametrize("always_write_timestamps", [False, True])
def test_stub_conversion_inspects_clean(tmp_path, always_write_timestamps):
    from nwbinspector import Importance, inspect_nwbfile
    from pynwb import NWBHDF5IO

    from neuroconv.datainterfaces import NatusRecordingInterface

    interface = NatusRecordingInterface(folder_path=session_dir(PRUNED))
    recording = interface.recording_extractor
    n_seg = recording.get_num_segments()
    assert n_seg == 24
    metadata = caller_metadata(interface)

    nwbfile_path = tmp_path / "natus_stub.nwb"
    interface.run_conversion(
        nwbfile_path=nwbfile_path,
        metadata=metadata,
        overwrite=True,
        stub_test=True,
        always_write_timestamps=always_write_timestamps,
    )

    with NWBHDF5IO(nwbfile_path, mode="r") as io:
        nwbfile = io.read()
        assert nwbfile.session_start_time == metadata["NWBFile"]["session_start_time"]
        assert nwbfile.subject.subject_id == "MSEL00000"
        series_names = sorted(name for name in nwbfile.acquisition if name.startswith("ElectricalSeries"))
        assert series_names == [f"ElectricalSeries{i:02d}" for i in range(n_seg)]
        assert len(nwbfile.electrodes) == recording.get_num_channels()
        assert set(nwbfile.electrode_groups) == set(recording.get_property("group_name"))
        assert list(nwbfile.electrodes["channel_name"][:]) == list(recording.get_property("channel_name"))

        previous_end = -np.inf
        for seg, name in enumerate(series_names):
            es = nwbfile.acquisition[name]
            data = es.data[:]
            n = min(100, recording.get_num_samples(seg))
            assert data.shape == (n, recording.get_num_channels())
            np.testing.assert_array_equal(data, recording.get_traces(segment_index=seg, start_frame=0, end_frame=n))
            # volts per count: a scalar conversion when every channel has the same gain
            channel_conversion = es.channel_conversion[:] if es.channel_conversion is not None else 1.0
            np.testing.assert_allclose(
                np.broadcast_to(channel_conversion * es.conversion, (recording.get_num_channels(),)),
                recording.get_channel_gains() * 1e-6,
            )
            t_start = recording.get_start_time(seg)
            if always_write_timestamps:
                assert es.rate is None
                timestamps = es.timestamps[:]
                assert timestamps.size == n
                np.testing.assert_allclose(timestamps, t_start + np.arange(n) / recording.get_sampling_frequency())
                assert timestamps[0] > previous_end  # the segments follow each other, gaps preserved
                previous_end = timestamps[-1]
            else:
                assert es.timestamps is None
                assert es.rate == recording.get_sampling_frequency()
                assert es.starting_time == pytest.approx(t_start)
        # the session starts with the first segment
        first = nwbfile.acquisition[series_names[0]]
        first_time = first.timestamps[0] if always_write_timestamps else first.starting_time
        assert first_time == 0.0

    messages = list(inspect_nwbfile(nwbfile_path=nwbfile_path))
    for message in messages:
        print(message.importance.name, message.check_function_name, message.object_name, message.message)
    serious = [m for m in messages if m.importance in (Importance.ERROR, Importance.CRITICAL, Importance.PYNWB_VALIDATION)]
    assert serious == []


def test_rate_change_session_streams(tmp_path):
    """c0fe2d81 has 256 Hz and 512 Hz files: one interface per stream."""
    from neuroconv.datainterfaces import NatusRecordingInterface

    if not DEV_ROOT:
        pytest.skip("NATUS_DEV_ROOT not set")
    d = session_dir(RATE_CHANGE_SESSION, root=DEV_ROOT)
    stream_names = NatusRecordingInterface.get_stream_names(folder_path=d)
    assert len(stream_names) == 2

    with pytest.raises(ValueError):
        NatusRecordingInterface(folder_path=d)

    rates = set()
    session_start_times = set()
    for stream_name in stream_names:
        interface = NatusRecordingInterface(folder_path=d, stream_name=stream_name)
        rates.add(interface.recording_extractor.get_sampling_frequency())
        session_start_times.add(interface.get_metadata()["NWBFile"]["session_start_time"])
        nwbfile = interface.create_nwbfile(metadata=caller_metadata(interface), stub_test=True)
        assert len(nwbfile.acquisition) == interface.recording_extractor.get_num_segments()
    assert rates == {256.0, 512.0}
    assert len(session_start_times) == 1  # both streams share the session's time base
