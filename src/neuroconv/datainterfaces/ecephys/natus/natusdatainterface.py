import re

import numpy as np
from pydantic import DirectoryPath

from ..baserecordingextractorinterface import BaseRecordingExtractorInterface
from ....utils import DeepDict

_TRAILING_NUMBER = re.compile(r"\d+$")


def _lead_name(channel_name: str) -> str:
    """The lead (electrode) a contact belongs to: its name without the trailing contact number.

    ``"LA1"`` and ``"LA12"`` give ``"LA"``; a name with no trailing number (``"EKG"``) is its own lead,
    and a name that is only a number keeps it.
    """
    name = str(channel_name).strip()
    lead = _TRAILING_NUMBER.sub("", name).rstrip(" -_")
    return lead or name or "ElectrodeGroup"


class NatusRecordingInterface(BaseRecordingExtractorInterface):
    """
    Data interface for converting Natus/XLTEK (NeuroWorks) EEG session folders.

    Uses :py:func:`~spikeinterface.extractors.read_natus` from SpikeInterface, which reads the
    session through :py:class:`neo.rawio.NatusRawIO` (``pip install neo[natus]``).

    A session recorded at several sample rates has one stream per rate; pick one with
    ``stream_name`` (see ``get_stream_names``). Each continuous piece of the stream is a segment and
    is written as its own ``ElectricalSeries`` (``ElectricalSeries00``, ``ElectricalSeries01``, ...),
    with its start time relative to the session start.

    Metadata comes from the recording itself, never from the folder name or patient fields: the
    session start time is the absolute (UTC) start of the first piece of the session, and the
    electrodes are grouped by lead, the channel name without its contact number (``LA1`` -> ``LA``).
    The subject is left to the caller.
    """

    display_name = "Natus Recording"
    keywords = BaseRecordingExtractorInterface.keywords + ("Natus", "XLTEK", "NeuroWorks", "EEG", "iEEG")
    associated_suffixes = (".erd", ".etc", ".stc", ".snc", ".ent")
    info = "Interface for Natus/XLTEK (NeuroWorks) EEG session folders."

    @classmethod
    def get_source_schema(cls) -> dict:
        source_schema = super().get_source_schema()
        source_schema["properties"]["folder_path"][
            "description"
        ] = 'Path to the Natus session folder holding the ".erd", ".etc", ".stc", ".snc" and ".ent" files.'
        return source_schema

    @classmethod
    def get_stream_names(cls, folder_path: DirectoryPath) -> list[str]:
        """
        Get the names of the streams in a Natus session folder, one per sample rate.

        Parameters
        ----------
        folder_path : DirectoryPath
            The session folder.

        Returns
        -------
        list of str
            The stream names, such as ``"512 Hz (46 ch)"``.
        """
        from spikeinterface.extractors.extractor_classes import NatusRecordingExtractor

        stream_names, _ = NatusRecordingExtractor.get_streams(folder_path=folder_path)
        return stream_names

    @classmethod
    def get_extractor_class(cls):
        from spikeinterface.extractors.extractor_classes import NatusRecordingExtractor

        return NatusRecordingExtractor

    def __init__(
        self,
        folder_path: DirectoryPath,
        *,
        stream_name: str | None = None,
        verbose: bool = False,
        es_key: str | None = None,
        metadata_key: str | None = None,
    ):
        """
        Load and prepare a Natus session for conversion.

        Parameters
        ----------
        folder_path : DirectoryPath
            The session folder.
        stream_name : str, optional
            The stream (sample rate) to read, as returned by ``get_stream_names``. Needed only when the
            session was recorded at more than one rate.
        verbose : bool, default: False
        es_key : str, optional
            Deprecated. Use ``metadata_key`` instead.
        metadata_key : str, optional
            Key that indexes this interface's entries in the dict-based metadata. Defaults to
            ``"natus_recording"``.
        """
        super().__init__(
            folder_path=folder_path,
            stream_name=stream_name,
            verbose=verbose,
            es_key=es_key,
            metadata_key=metadata_key,
        )

        if metadata_key is None:
            self.metadata_key = "natus_recording"

        recording = self.recording_extractor
        neo_reader = recording.neo_reader
        self.session_start_time = neo_reader.raw_annotations["blocks"][recording.block_index]["rec_datetime"]
        self.stream_name = recording.stream_name
        self.sampling_frequency = recording.get_sampling_frequency()

        # properties of object or bool dtype are not supported by NWB
        for key in recording.get_property_keys():
            value = recording.get_property(key)
            if value.dtype == object or value.dtype == np.bool_:
                recording.set_property(key, np.asarray(value, dtype=str))

        channel_names = recording.get_property("channel_name")
        if channel_names is None:
            channel_names = recording.get_channel_ids()
        leads = [_lead_name(name) for name in channel_names]
        lead_order = list(dict.fromkeys(leads))  # in order of first appearance
        recording.set_channel_groups([lead_order.index(lead) for lead in leads])
        recording.set_property("group_name", np.array(leads))

    def get_metadata(self, *, use_new_metadata_format: bool = True) -> DeepDict:
        nwbfile_metadata = dict(session_start_time=self.session_start_time)
        series_description = (
            f"EEG from a Natus/XLTEK (NeuroWorks) session, the {self.sampling_frequency:g} Hz stream; "
            "one series per continuous piece of recording."
        )
        device = dict(name="NatusAmplifier", description="Natus/XLTEK amplifier (NeuroWorks).")

        from ....tools.spikeinterface.spikeinterface import _get_group_name

        lead_names = sorted(set(_get_group_name(recording=self.recording_extractor).tolist()))

        if use_new_metadata_format:
            metadata = super().get_metadata(use_new_metadata_format=True)
            metadata["NWBFile"].update(nwbfile_metadata)
            metadata["Ecephys"]["ElectricalSeries"][self.metadata_key]["description"] = series_description

            device_metadata_key = "natus_device"
            metadata["Devices"] = {device_metadata_key: device}
            metadata["Ecephys"]["ElectrodeGroups"] = {
                lead: dict(
                    name=lead,
                    description=f"Contacts of lead {lead}.",
                    location="unknown",
                    device_metadata_key=device_metadata_key,
                )
                for lead in lead_names
            }
            return metadata

        metadata = super().get_metadata(use_new_metadata_format=False)
        metadata["NWBFile"].update(nwbfile_metadata)
        # replace (not merge) the default device and groups
        metadata["Ecephys"]["Device"] = [device]
        metadata["Ecephys"]["ElectrodeGroup"] = [
            dict(name=lead, description=f"Contacts of lead {lead}.", location="unknown", device=device["name"])
            for lead in lead_names
        ]
        if self.es_key is not None:
            metadata["Ecephys"][self.es_key] = dict(name=self.es_key, description=series_description)
        return metadata
