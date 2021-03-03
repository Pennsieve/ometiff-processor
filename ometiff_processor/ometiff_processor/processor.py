import json
import os
import re

import PIL.Image
import bioformats as bf
import bioformats.omexml as ome
import boto3
import numpy as np
from base_processor.imaging import utils
from botocore.client import Config
from javabridge.jutil import JavaException

from base_image_microscopy_processor import BaseMicroscopyImageProcessor


class OMETIFFOutputFile(object):
    def __init__(self, *args, **kwargs):
        self.img_dimensions = {}
        self.num_dimensions = -1
        self.file_path = None
        self.view_path = None

        self.img_data = kwargs.get('img_data', None)
        self.img_data_dtype = kwargs.get('img_data_dtype', None)
        self.metadata = kwargs.get('metadata', {})

        self.img_rdr = kwargs.get('img_rdr', None)
        self.DimensionOrder = kwargs.get('DimensionOrder', None)
        self.ImageCount = kwargs.get('ImageCount', -1)
        self.SizeX = kwargs.get('SizeX', -1)
        self.SizeY = kwargs.get('SizeY', -1)
        self.SizeZ = kwargs.get('SizeZ', -1)
        self.SizeC = kwargs.get('SizeC', -1)
        self.SizeT = kwargs.get('SizeT', -1)
        self.PixelType = kwargs.get('PixelType', None)
        self.RGBChannelCount = kwargs.get('RGBChannelCount', -1)

        self.isRGB = kwargs.get('isRGB', False)
        self.RGBDimension = kwargs.get('RGBDimension', -1)
        self.hasTimeDimension = kwargs.get('hasTimeDimension', False)
        self.TimeDimension = kwargs.get('TimeDimension', -1)

        self.view_format = 'dzi'
        self.optimize = kwargs.get('optimize', False)
        self.tile_size = kwargs.get('tile_size', 128)
        self.tile_overlap = kwargs.get('tile_overlap', 0)
        self.tile_format = kwargs.get('tile_format', "png")
        self.image_quality = kwargs.get('image_quality', 1.0)
        self.resize_filter = kwargs.get('resize_filter', "bicubic")

    def _load_and_save_assets(self, i_xyzct, n_xyzct):
        """Load image using BioFormats"""

        # Parse parallelization arguments
        i_x, i_y, i_z, i_c, i_t = i_xyzct
        n_x, n_y, n_z, n_c, n_t = n_xyzct

        if i_x == -1 or n_x == -1:
            i_x = 0
            n_x = 1
        if i_y == -1 or n_y == -1:
            i_y = 0
            n_y = 1
        if i_z == -1 or n_z == -1:
            i_z = 0
            n_z = 1
        if i_c == -1 or n_c == -1:
            i_c = 0
            n_c = 1
        if i_t == -1 or n_t == -1:
            i_t = 0
            n_t = 1

        # Make view directory
        if not os.path.exists('%s-zoomed' % os.path.basename(self.file_path)):
            os.makedirs('%s-zoomed' % os.path.basename(self.file_path))
        asset_format = self.view_format

        with bf.ImageReader(path=self.file_path) as rdr:
            if ''.join(self.DimensionOrder) == 'XYCZT' and self.isRGB:
                for z in range(int(self.SizeZ / float(n_z) * i_z), int(self.SizeZ / float(n_z) * (i_z + 1))):
                    for t in range(int(self.SizeT / float(n_t) * i_t), int(self.SizeT / float(n_t) * (i_t + 1))):
                        # Get image data matrix
                        # Y and X flipped because image
                        self.img_data = np.zeros((self.SizeY, self.SizeX, self.SizeC))
                        self.img_data = rdr.read(z=z, t=t)

                        # Convert type of image in order to save in deep-zoom
                        self.img_data = self._convert_image_data_type(self.img_data)
                        image = PIL.Image.fromarray(self.img_data, "RGB")
                        self._save_view(image, z=z, t=t, is_rgb=True, asset_format=asset_format)
            else:
                for z in range(int(self.SizeZ / float(n_z) * i_z), int(self.SizeZ / float(n_z) * (i_z + 1))):
                    for t in range(int(self.SizeT / float(n_t) * i_t), int(self.SizeT / float(n_t) * (i_t + 1))):
                        for c in range(int(self.SizeC / float(n_c) * i_c), int(self.SizeC / float(n_c) * (i_c + 1))):
                            # Get image data matrix
                            # Y and X flipped because image
                            self.img_data = np.zeros((self.SizeY, self.SizeX))
                            self.img_data = rdr.read(index=t + c * self.SizeT + z * self.SizeC * self.SizeT)

                            # Convert type of image in order to save in deep-zoom
                            self.img_data = self._convert_image_data_type(img_data=self.img_data)
                            image = PIL.Image.fromarray(self.img_data)
                            self._save_view(image, z=z, c=c, t=t, is_rgb=True, asset_format=asset_format)

        self.view_path = os.path.join(os.getcwd(), '%s-zoomed' % os.path.basename(self.file_path))
        return

    def _convert_image_data_type(self, img_data=None, format=np.uint8):
        """Convert image data matrix datatype to another format"""

        # Check correct data type format to convert
        if format in np.typeDict.values():
            pass
        elif type(format) == str:
            try:
                format = np.dtype(format)
            except Exception:
                raise TypeError("Image data type to convert to should be of type numpy.dtype")
        else:
            raise TypeError("Image data type to convert to should be of type numpy.dtype")

        if img_data is None:
            # Convert only if image data type is not in desired data type format
            if self.img_data_dtype != format:
                self.img_data = utils.convert_image_data_type(self.img_data, format)
                self.img_data_dtype = format
            return
        else:
            img_data = utils.convert_image_data_type(img_data, format)
            return img_data

    def _save_view(self, image, z=None, c=None, t=None, is_rgb=False, asset_format='dzi'):
        """"""

        if is_rgb:
            # Save asset in appropriate format
            filename = os.path.join(
                '%s-zoomed' % os.path.basename(self.file_path),
                'dim_Z_slice_{slice_z}_dim_T_slice_{slice_t}.{fmt}'.format(
                    slice_z=z, slice_t=t, fmt=asset_format)
            )

            utils.save_asset(
                image,
                asset_format,
                filename,
                optimize=self.optimize, tile_size=self.tile_size,
                tile_overlap=self.tile_overlap, tile_format=self.tile_format,
                image_quality=self.image_quality, resize_filter=self.resize_filter
            )
            # Save thumbnail
            if asset_format == 'dzi':
                timage = image.copy()
                timage.thumbnail((200, 200), PIL.Image.ANTIALIAS)
                timage.save(
                    os.path.join(
                        '%s-zoomed' % os.path.basename(self.file_path),
                        'dim_Z_slice_{slice_z}_dim_T_slice_{slice_t}_thumbnail.png'.format(
                            slice_z=z,
                            slice_t=t
                        )
                    )
                )
                # Create large thumbnail
                timage = image.copy()
                timage.thumbnail((1000, 1000), PIL.Image.ANTIALIAS)
                timage.save(
                    os.path.join(
                        '%s-zoomed' % os.path.basename(self.file_path),
                        'dim_Z_slice_{slice_z}_dim_T_slice_{slice_t}_large_thumbnail.png'.format(
                            slice_z=z,
                            slice_t=t
                        )
                    )
                )
        else:
            # Save asset in appropriate format
            filename = os.path.join(
                '%s-zoomed' % os.path.basename(self.file_path),
                'dim_Z_slice_{slice_z}_dim_C_slice_{slice_c}_dim_T_slice_{slice_t}.{fmt}'.format(
                    slice_z=z, slice_c=c, slice_t=t, fmt=asset_format)
            )

            utils.save_asset(
                image,
                asset_format,
                filename,
                optimize=self.optimize, tile_size=self.tile_size,
                tile_overlap=self.tile_overlap, tile_format=self.tile_format,
                image_quality=self.image_quality, resize_filter=self.resize_filter
            )
            # Save thumbnail
            if asset_format == 'dzi':
                timage = image.copy()
                timage.thumbnail((200, 200), PIL.Image.ANTIALIAS)
                timage.save(
                    os.path.join(
                        '%s-zoomed' % os.path.basename(self.file_path),
                        'dim_Z_slice_{slice_z}_dim_C_slice_{slice_c}_'
                        'dim_T_slice_{slice_t}_thumbnail.png'.format(
                            slice_z=z,
                            slice_c=c,
                            slice_t=t
                        )
                    )
                )
                # Create large thumbnail
                timage = image.copy()
                timage.thumbnail((1000, 1000), PIL.Image.ANTIALIAS)
                timage.save(
                    os.path.join(
                        '%s-zoomed' % os.path.basename(self.file_path),
                        'dim_Z_slice_{slice_z}_dim_C_slice_{slice_c}_'
                        'dim_T_slice_{slice_t}_large_thumbnail.png'.format(
                            slice_z=z,
                            slice_c=c,
                            slice_t=t
                        )
                    )
                )
        return

    @property
    def file_size(self):
        """Return file size"""
        return os.path.getsize(self.file_path)

    @property
    def view_size(self):
        """"Return size of exploded view assets"""
        return os.path.getsize(self.view_path)

    def get_view_asset_dict(self, storage_bucket, upload_key):
        """Generate JSON dictionary for making View mutations"""
        upload_key = upload_key.rstrip('/')
        json_dict = {
            "bucket": storage_bucket,
            "key": upload_key,
            "type": "View",
            "size": self.view_size,
            "fileType": "Image"
        }
        return json_dict

    def get_dim_assignment(self):
        """Retrieve inferred dimension assignment based on number of dimensions and length of each dimension. """
        return self.DimensionOrder

    def set_img_properties(self, asset_format='dzi'):
        """Create and assign properties for the image"""

        self.view_format = asset_format

        # Load image using BioFormats
        try:
            ImageReader = bf.formatreader.make_image_reader_class()
            self.img_rdr = ImageReader()
            self.img_rdr.setId(self.file_path)
        except JavaException:
            raise

        # Get image data details
        self.DimensionOrder = list(self.img_rdr.getDimensionOrder())
        self.ImageCount = self.img_rdr.getImageCount()
        self.SizeX = self.img_rdr.getSizeX()
        self.SizeY = self.img_rdr.getSizeY()
        self.SizeC = self.img_rdr.getSizeC()
        self.SizeT = self.img_rdr.getSizeT()
        self.SizeZ = self.img_rdr.getSizeZ()
        self.PixelType = self.img_rdr.getPixelType()
        self.RGBChannelCount = self.img_rdr.getRGBChannelCount()
        self.isRGB = self.img_rdr.isRGB()

        self.RGBDimension = 3
        self.hasTimeDimension = True
        self.TimeDimension = 4

        # Set data type #TODO: For some reason, rdr.read returns np.float64 instead of specified PixelType
        self.PixelType = 6
        if self.PixelType == 0:  # int8
            # self.img_data = self.img_data.astype(np.int8)
            self.img_data_dtype = np.int8
        elif self.PixelType == 1:  # uint8
            # self.img_data = self.img_data.astype(np.uint8)
            self.img_data_dtype = np.uint8
        elif self.PixelType == 2:  # int16
            # self.img_data = self.img_data.astype(np.int16)
            self.img_data_dtype = np.int16
        elif self.PixelType == 3:  # uint16
            # self.img_data = self.img_data.astype(np.uint16)
            self.img_data_dtype = np.uint16
        elif self.PixelType == 4:  # int32
            # self.img_data = self.img_data.astype(np.int32)
            self.img_data_dtype = np.int32
        elif self.PixelType == 5:  # uint32
            # self.img_data = self.img_data.astype(np.uint32)
            self.img_data_dtype = np.uint32
        elif self.PixelType == 6:  # float
            # self.img_data = self.img_data.astype(np.float)
            self.img_data_dtype = np.float
        elif self.PixelType == 7:  # double
            # self.img_data = self.img_data.astype(np.double)
            self.img_data_dtype = np.double

        # Set number of dimensions of image matrix
        self.num_dimensions = 5
        self.img_data_shape = [self.SizeX, self.SizeY, self.SizeZ, self.SizeC, self.SizeT]

        dim_assignment = list('XYZCT')  # Force assignment

        self.img_dimensions['filename'] = os.path.basename(self.file_path)
        self.img_dimensions['num_dimensions'] = self.num_dimensions
        self.img_dimensions['isColorImage'] = False
        self.img_dimensions['dimensions'] = {}

        for dim in range(self.num_dimensions):
            self.img_dimensions['dimensions'][dim] = {}
            self.img_dimensions['dimensions'][dim]["assignment"] = dim_assignment[dim]
            self.img_dimensions['dimensions'][dim]["length"] = self.img_data_shape[dim]
            self.img_dimensions['dimensions'][dim]["resolution"] = -1
            self.img_dimensions['dimensions'][dim]["units"] = "um"
            if dim_assignment[dim] == 'C' and self.isRGB:
                self.RGBDimension = dim
            if dim_assignment[dim] == 'T':
                self.hasTimeDimension = True
                self.TimeDimension = dim
        self.img_dimensions['isColorImage'] = self.isRGB

        # Get metadata
        metadata = bf.get_omexml_metadata(self.file_path).encode('utf-8')

        # Get resolution
        resx_node = 'PhysicalSizeX="'
        resy_node = 'PhysicalSizeY="'
        try:
            start = metadata.index(resx_node) + len(resx_node)
            end = metadata.index('"', start)
            self.img_dimensions['dimensions'][0]["resolution"] = float(metadata[start:end])
        except ValueError:
            self.img_dimensions['dimensions'][0]["resolution"] = -1
        try:
            start = metadata.index(resy_node) + len(resy_node)
            end = metadata.index('"', start)
            self.img_dimensions['dimensions'][1]["resolution"] = float(metadata[start:end])
        except ValueError:
            self.img_dimensions['dimensions'][1]["resolution"] = -1

        try:
            self.metadata = bf.omexml.OMEXML(metadata.decode('utf-8'))
        except Exception:
            # Raise an exception only if we file ends in .ome.tif/.ome.tiff since we know it should be OME compatible
            if self.file_path.endswith('.ome.tiff') or self.file_path.endswith('.ome.tif'):
                raise TypeError("XML metadata decoding failed.")
        # self.metadata = javabridge.jdictionary_to_string_dictionary(self.img_rdr.getMetadata())
        return

    def load_and_save_assets(self, ometiff_file_path, i_xyzct=(-1, -1, -1, -1, -1), n_xyzct=(-1, -1, -1, -1, -1),
                             asset_format='dzi'):
        """Load image and generate view assets"""
        # Set file path
        self.file_path = ometiff_file_path

        # Set image properties
        self.set_img_properties(asset_format=asset_format)

        # Load image
        self._load_and_save_assets(i_xyzct=i_xyzct, n_xyzct=n_xyzct)

class OMETIFFProcessor(BaseMicroscopyImageProcessor):
    required_inputs = ['file']

    def __init__(self, *args, **kwargs):
        super(OMETIFFProcessor, self).__init__(*args, **kwargs)
        self.session = boto3.session.Session()
        self.s3_client = self.session.client('s3', config=Config(signature_version='s3v4'))

        self.file = self.inputs.get('file')

        self.upload_key = None
        try:
            self.optimize = utils.str2bool(self.inputs.get('optimize_view'))
        except AttributeError:
            self.optimize = False

        try:
            self.tile_size = int(self.inputs.get('tile_size'))
        except (ValueError, KeyError, TypeError) as e:
            self.tile_size = 128

        try:
            self.tile_overlap = int(self.inputs.get('tile_overlap'))
        except (ValueError, KeyError, TypeError) as e:
            self.tile_overlap = 0

        try:
            self.tile_format = self.inputs.get('tile_format')
            if self.tile_format is None:
                self.tile_format = "png"
        except KeyError:
            self.tile_format = "png"

        try:
            self.image_quality = float(self.inputs.get('image_quality'))
        except (ValueError, KeyError, TypeError) as e:
            self.image_quality = 1.0

        try:
            self.resize_filter = self.inputs.get('resize_filter')
        except KeyError:
            self.resize_filter = "bicubic"

    def load_and_save(self, i_xyzct=(-1, -1, -1, -1, -1), n_xyzct=(-1, -1, -1, -1, -1)):
        """Run load_image for all output files"""
        if os.path.isfile(self.file):
            output_file = OMETIFFOutputFile(optimize=self.optimize, tile_size=self.tile_size,
                                            tile_overlap=self.tile_overlap, tile_format=self.tile_format,
                                            image_quality=self.image_quality, resize_filter=self.resize_filter)
            output_file.load_and_save_assets(self.file, i_xyzct=i_xyzct, n_xyzct=n_xyzct)
            self.outputs.append(output_file)

    def task(self):
        """Run main task which will load and save view assets for all OME-TIFF output files deocded"""
        # TODO: Add capability for BigTIFF file formats as well as ome.tif (in addition to ome.tiff)

        # self._load_image()
        self.LOGGER.info('Got inputs {}'.format(self.inputs))

        # Get sub_region index
        try:
            sub_region = self.inputs['sub_region_file']
            sub_region_regex = r'sub_' \
                               r'x_([0-9]+)_([0-9]+)_' \
                               r'y_([0-9]+)_([0-9]+)_' \
                               r'z_([0-9]+)_([0-9]+)_' \
                               r'c_([0-9]+)_([0-9]+)_' \
                               r't_([0-9]+)_([0-9]+).txt'
            i_x = int(re.match(re.compile(sub_region_regex), sub_region).groups()[0])
            n_x = int(re.match(re.compile(sub_region_regex), sub_region).groups()[1])
            i_y = int(re.match(re.compile(sub_region_regex), sub_region).groups()[2])
            n_y = int(re.match(re.compile(sub_region_regex), sub_region).groups()[3])
            i_z = int(re.match(re.compile(sub_region_regex), sub_region).groups()[4])
            n_z = int(re.match(re.compile(sub_region_regex), sub_region).groups()[5])
            i_c = int(re.match(re.compile(sub_region_regex), sub_region).groups()[6])
            n_c = int(re.match(re.compile(sub_region_regex), sub_region).groups()[7])
            i_t = int(re.match(re.compile(sub_region_regex), sub_region).groups()[8])
            n_t = int(re.match(re.compile(sub_region_regex), sub_region).groups()[9])
        except (KeyError, IndexError, AttributeError):
            i_x = -1
            n_x = -1
            i_y = -1
            n_y = -1
            i_z = -1
            n_z = -1
            i_c = -1
            n_c = -1
            i_t = -1
            n_t = -1

        # Load and save view images
        self.load_and_save(i_xyzct=(i_x, i_y, i_z, i_c, i_t), n_xyzct=(n_x, n_y, n_z, n_c, n_t))

        if os.path.isfile(self.file):
            # Output file is just the one and only file in outputs
            output_file = self.outputs[0]

            # Save dimensions object as JSON in view/ directory (for now)
            with open(os.path.join('%s-zoomed' % os.path.basename(self.file), 'dimensions.json'), 'w') as fp:
                json.dump(output_file.img_dimensions, fp)

            # Create create-asset JSON object file called view_asset_info.json
            self.upload_key = os.path.join(
                self.settings.storage_directory,
                os.path.basename(output_file.file_path) + '-zoomed'
            )
            with open('view_asset_info.json', 'w') as fp:
                json.dump(output_file.get_view_asset_dict(
                    self.settings.storage_bucket,
                    self.upload_key
                ), fp)

            # Generate properties metadata.json metadata
            metadata = []
            img_dimensions = self.outputs[0].img_dimensions
            for dim in range(self.outputs[0].num_dimensions):
                for property_key_suffix in ["assignment", "length", "resolution", "units"]:
                    # Initialize property
                    property = {}

                    # Set property key and value
                    property_key = 'dimensions.%i.%s' % (dim, property_key_suffix)
                    property_value = str(img_dimensions['dimensions'][dim][property_key_suffix])

                    # Create property instance
                    property["key"] = property_key
                    property["value"] = property_value
                    property["dataType"] = "String"
                    property["category"] = "Blackfynn"
                    property["fixed"] = False
                    property["hidden"] = False
                    metadata.append(property)
            with open('metadata.json', 'w') as fp:
                json.dump(metadata, fp)

            ## Hack for backwards compatibility for DeepZoom viewer
            if (i_x, i_y, i_z, i_c, i_t) == (-1, -1, -1, -1, -1) or (i_x, i_y, i_z, i_c, i_t) == (0, 0, 0, 0, 0):
                with bf.ImageReader(path=self.outputs[0].file_path) as rdr:
                    if ''.join(self.outputs[0].DimensionOrder) == 'XYCZT' and self.outputs[0].isRGB:
                        img_data = np.zeros((self.outputs[0].SizeY, self.outputs[0].SizeX, 3))
                        for z in range(self.outputs[0].SizeZ):
                            for t in range(self.outputs[0].SizeT):
                                img_data_slice = rdr.read(z=z, t=t)
                                img_data = np.max((img_data, img_data_slice), axis=0)
                    else:
                        img_data = np.zeros((self.outputs[0].SizeY, self.outputs[0].SizeX, self.outputs[0].SizeC))
                        for z in range(self.outputs[0].SizeZ):
                            for t in range(self.outputs[0].SizeT):
                                for c in range(self.outputs[0].SizeC):
                                    img_data_slice = rdr.read(
                                        index=t + c * self.outputs[0].SizeT + z * self.outputs[0].SizeC * self.outputs[
                                            0].SizeT)
                                    img_data[:, :, c] = np.max((img_data[:, :, c].squeeze(), img_data_slice), axis=0)

                mip_img_data = np.zeros((self.outputs[0].SizeY, self.outputs[0].SizeX, 3))
                if self.outputs[0].SizeC == 1:
                    mip_img_data[:, :, :1] = img_data
                if self.outputs[0].SizeC == 2:
                    mip_img_data[:, :, :2] = img_data
                if self.outputs[0].SizeC == 3:
                    mip_img_data = img_data
                if self.outputs[0].SizeC > 3:
                    mip_img_data = img_data[:, :, :3]

                mip_img_data = utils.convert_image_data_type(mip_img_data, np.uint8)
                image = PIL.Image.fromarray(mip_img_data, "RGB")

                # Save asset as slide.dzi for backwards compatibility
                filename = os.path.join(
                    '%s-zoomed' % os.path.basename(self.outputs[0].file_path),
                    'slide.dzi'
                )
                utils.save_asset(
                    image,
                    'dzi',
                    filename,
                    optimize=self.optimize, tile_size=self.tile_size,
                    tile_overlap=self.tile_overlap, tile_format=self.tile_format,
                    image_quality=self.image_quality, resize_filter=self.resize_filter
                )

                # Save thumbnail
                timage = image.copy()
                timage.thumbnail((200, 200), PIL.Image.ANTIALIAS)
                timage.save(
                    os.path.join(
                        '%s-zoomed' % os.path.basename(self.outputs[0].file_path),
                        'slide_thumbnail.png'
                    )
                )
                # Create large thumbnail
                timage = image.copy()
                timage.thumbnail((1000, 1000), PIL.Image.ANTIALIAS)
                timage.save(
                    os.path.join(
                        '%s-zoomed' % os.path.basename(self.outputs[0].file_path),
                        'slide_large_thumbnail.png'
                    )
                )
