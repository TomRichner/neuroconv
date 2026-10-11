Natus/XLTEK (NeuroWorks) data conversion
----------------------------------------

Install NeuroConv with the additional dependencies necessary for reading Natus data (the reader
needs ``neo.rawio.NatusRawIO`` and Numba).

.. code-block:: bash

    pip install "neuroconv[natus]"

A Natus session folder holds the ``.erd`` sample files with their ``.etc`` indexes and the
session-wide ``.stc``, ``.snc`` and ``.ent`` files. The
:py:class:`~neuroconv.datainterfaces.ecephys.natus.natusdatainterface.NatusRecordingInterface`
reads one sample rate (stream) at a time: a session recorded at several rates has one stream per
rate, listed by ``NatusRecordingInterface.get_stream_names(folder_path=folder_path)``. Each
continuous piece of recording becomes its own ``ElectricalSeries``, timed from the start of the
session. Electrodes are grouped by lead (``LA1``, ``LA2``, ... form the group ``LA``).

The session start time comes from the recording; no patient information is read, so the subject
is yours to add.

.. code-block:: python

    from neuroconv.datainterfaces import NatusRecordingInterface

    folder_path = "path/to/natus_session_folder"
    stream_names = NatusRecordingInterface.get_stream_names(folder_path=folder_path)
    interface = NatusRecordingInterface(folder_path=folder_path, stream_name=stream_names[0])

    metadata = interface.get_metadata()
    metadata["Subject"] = dict(subject_id="subject1", species="Homo sapiens", sex="U", age="P30Y")

    interface.run_conversion(nwbfile_path="natus.nwb", metadata=metadata, overwrite=True)
