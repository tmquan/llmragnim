from typing import Iterator
import json
import zipfile
import io
import os

from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Union,
)

from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseBlobParser 
from langchain_community.document_loaders.pdf import BasePDFLoader
from langchain_community.document_loaders.blob_loaders import Blob


class AdobePDFParser(BaseBlobParser):
    """Loads a document using the Adobe PDF Services API."""

    def __init__(self, client_id: str, client_secret: str, mode: str = "chunks", embed_figures: bool = True):
        from adobe.pdfservices.operation.auth.service_principal_credentials import ServicePrincipalCredentials
        from adobe.pdfservices.operation.pdf_services import PDFServices

        self.credentials = ServicePrincipalCredentials(client_id=client_id, client_secret=client_secret)
        self.pdf_services = PDFServices(credentials=self.credentials)
        self.mode = mode
        self.embed_figures = embed_figures
        assert self.mode in ["json", "chunks", "data"]

    def _encode_image_as_base64(self, image_path: str, archive: zipfile.ZipFile) -> str:
        """Encode an image as a base64 string."""
        import base64

        with archive.open(image_path, "r") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def _load_table_as_markdown(self, file_path: str, archive: zipfile.ZipFile) -> str:
        import pandas as pd

        with archive.open(file_path) as file:
            df = pd.read_csv(filepath_or_buffer=file, on_bad_lines="warn")
        return df.to_markdown()

    def _generate_docs_chunks(self, json_data: dict, data: zipfile.ZipFile) -> Iterator[Document]:
        headers = []
        current_paragraphs = []
        header_page = None
        paragraph_page = None
        last_header_level = None
        figures = {}

        def yield_chunk():
            if current_paragraphs:
                merged_paragraphs = "".join([p for p in current_paragraphs]).strip()
                yield Document(
                    page_content=merged_paragraphs,
                    metadata={
                        "headers": headers.copy(),
                        "page": header_page + 1 if header_page else paragraph_page + 1,
                        "figures": figures,
                    },
                )
                current_paragraphs.clear()
                figures.clear()

        for element in json_data["elements"]:
            text = element.get("Text", "")
            path = element.get("Path", "")
            page = element.get("Page", None)

            if "/H" in path and text:
                header_level = int(path.split("/H")[-1][0])

                if "Figure" in path:
                    continue

                if last_header_level and last_header_level < header_level:
                    yield from yield_chunk()
                    headers.append(text)

                else:
                    yield from yield_chunk()
                    headers = headers[: header_level - 1]
                    headers.append(text)

                last_header_level = header_level
                header_page = page
            else:
                file_paths = element.get("filePaths", [])
                paragraph_page = element.get("Page")

                if "Figure" in path and file_paths and self.embed_figures:
                    for file_path in file_paths:
                        if path not in figures:
                            figures[path] = []
                        figures[path].append(self._encode_image_as_base64(image_path=file_path, archive=data))
                        current_paragraphs.append(f"{{{file_path}}}\n")

                elif "Figure" in path and text and not self.embed_figures:
                    current_paragraphs.append(text + "\n")

                elif "Table" in path and file_paths:
                    for file_path in file_paths:
                        table = self._load_table_as_markdown(file_path=file_path, archive=data)
                        current_paragraphs.append(table + "\n")

                elif "Lbl" in path:
                    current_paragraphs.append(text + " ")

                elif "Table" not in path:
                    current_paragraphs.append(text + "\n")
        yield from yield_chunk()

    def _generate_docs_data(self, archive: zipfile.ZipFile) -> Iterator[Document]:
        from mimetypes import guess_type

        for name in archive.namelist():
            mime_type, _ = guess_type(name)
            if mime_type == "image/png":
                figure_base64 = self._encode_image_as_base64(image_path=name, archive=archive)
                yield Document(page_content=figure_base64, metadata={"content_type": "base64"})
            elif mime_type == "text/csv":
                table_md = self._load_table_as_markdown(file_path=name, archive=archive)
                yield Document(page_content=table_md, metadata={"content_type": "markdown"})

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Lazily parse the blob."""
        from adobe.pdfservices.operation.io.cloud_asset import CloudAsset
        from adobe.pdfservices.operation.io.stream_asset import StreamAsset
        from adobe.pdfservices.operation.pdf_services_media_type import PDFServicesMediaType
        from adobe.pdfservices.operation.pdfjobs.jobs.extract_pdf_job import ExtractPDFJob
        from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_element_type import ExtractElementType
        from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_pdf_params import ExtractPDFParams
        from adobe.pdfservices.operation.pdfjobs.result.extract_pdf_result import ExtractPDFResult
        from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.table_structure_type import TableStructureType
        from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_renditions_element_type import (
            ExtractRenditionsElementType,
        )

        blob_bytes = blob.as_bytes()
        import tempfile

        with tempfile.TemporaryFile() as temp_file:
            temp_file.write(blob_bytes)
            temp_file.seek(0)

            input_asset = self.pdf_services.upload(input_stream=temp_file, mime_type=PDFServicesMediaType.PDF)

            extract_pdf_params = ExtractPDFParams(
                elements_to_extract=[ExtractElementType.TEXT, ExtractElementType.TABLES],
                table_structure_type=TableStructureType.CSV,
                elements_to_extract_renditions=[ExtractRenditionsElementType.FIGURES],
            )

            extract_pdf_job = ExtractPDFJob(input_asset=input_asset, extract_pdf_params=extract_pdf_params)

            location = self.pdf_services.submit(extract_pdf_job)
            pdf_services_response = self.pdf_services.get_job_result(location, ExtractPDFResult)

            result_asset: CloudAsset = pdf_services_response.get_result().get_resource()
            stream_asset: StreamAsset = self.pdf_services.get_content(result_asset)
            zip_bytes = stream_asset.get_input_stream()

            with io.BytesIO(zip_bytes) as memory_zip:
                with zipfile.ZipFile(memory_zip, "r") as archive:
                    with archive.open("structuredData.json") as jsonentry:
                        jsondata = jsonentry.read()
                        data = json.loads(jsondata)
                        if self.mode == "json":
                            yield data
                        elif self.mode == "data":
                            yield from self._generate_docs_data(archive=archive)
                        elif self.mode == "chunks":
                            yield from self._generate_docs_chunks(json_data=data, data=archive)
                        else:
                            raise ValueError(f"Invalid mode: {self.mode}")

class AdobePDFLoader(BasePDFLoader):
    """Load PDF using pypdf into list of documents.

    Loader chunks by page and stores page numbers in metadata.
    """

    def __init__(
        self,
        file_path: str,
        password: Optional[Union[str, bytes]] = None,
        headers: Optional[Dict] = None,
        extract_images: bool = False,
        *,
        parser: AdobePDFParser
        # extraction_mode: str = "data",
        # extraction_kwargs: Optional[Dict] = None,
    ) -> None:
        """Initialize with a file path."""
        super().__init__(file_path, headers=headers)
        # self.parser = AdobePDFExtractionParser(
        #     client_id=os.getenv('ADOBE_CLIENT_ID'),
        #     client_secret=os.getenv('ADOBE_CLIENT_SECRET'),
        #     mode=extraction_mode
        # )
        self.parser = parser

    def lazy_load(
        self,
    ) -> Iterator[Document]:
        """Lazy load given path as pages."""
        if self.web_path:
            blob = Blob.from_data(open(self.file_path, "rb").read(), path=self.web_path)  # type: ignore[attr-defined]
        else:
            blob = Blob.from_path(self.file_path)  # type: ignore[attr-defined]
        yield from self.parser.parse(blob)