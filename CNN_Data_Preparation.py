# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 10:38:38 2021

@author: tanuj
"""

import ee

from numpy.random import seed
from numpy.random import rand

seed(10)
values = rand(5000)
print(values)

ee.Initialize()

patch = ee.Image("users/khanalnishant/osmlayers/Nepal/buildingRaster").rename("class")

new_patch = ee.Image("projects/servir-hkh/RLCMS/construction/HKH/osmdata/bldgRaster_10").rename("class")

boundary = ee.FeatureCollection('projects/servir-hkh/RLCMS/NEP/bnd/bnd_buff').geometry();


xs = 128

myList = ee.List.repeat(1, xs)
myLists = ee.List.repeat(myList, xs)
kernel = ee.Kernel.fixed(xs, xs,myLists)
outputBucket = 'tfworkshop'
folder = 'HKH/Tanuja/New_data8'

print(len(values))

#var polyFeature = ee.Feature(polygon, {foo: 42, bar: 'tart'});a

pt = ee.FeatureCollection(
    [ee.Feature(
        ee.Geometry.Point([84.019, 28.241]),
        {
          "system:index": "0"
        }),
    ee.Feature(
        ee.Geometry.Point([84.425, 27.688]),
        {
          "system:index": "1"
        }),
    ee.Feature(
        ee.Geometry.Point([81.58, 28.086]),
        {
          "system:index": "2"
        }),
    ee.Feature(
        ee.Geometry.Point([81.272, 29.147]),
        {
          "system:index": "3"
        }),
    ee.Feature(
        ee.Geometry.Point([80.361, 29.061])),
    ee.Feature(   
        ee.Geometry.Point([85.03, 27.435])),
    ee.Feature(   
        ee.Geometry.Point([85.524, 27.63])), 
    ee.Feature(   
        ee.Geometry.Point([86.6227, 27.4204])),  
    ee.Feature(   
        ee.Geometry.Point([87.265, 26.461])),
    ee.Feature(   
        ee.Geometry.Point([87.881, 26.912]))])



def hello(feature):
  return feature.geometry().buffer(6000).bounds();
geom = pt.map(hello)


#cloud masking function of sentinnel data
def maskS2clouds(image):
  qa = image.select('QA60')

  # Bits 10 and 11 are clouds and cirrus, respectively.
  cloudBitMask = 1 << 10;
  cirrusBitMask = 1 << 11;

  # Both flags should be set to zero, indicating clear conditions.
  mask = qa.bitwiseAnd(cloudBitMask).eq(0)
  mask = (qa.bitwiseAnd(cirrusBitMask).eq(0))
  return image.updateMask(mask).divide(10000)

NDVI_bands = ['B8','B4'] 
NDWI_bands = ['B3','B8']   


"""
SWIR(Band11)-NIR(Band8)/ SWIR(Band11)+NIR(Band8)
NDWI = (B03 - B08) / (B03 + B08)
ndvi = B 8 − B 4 B 8 + B 4

"""

for n in range(1,5000,1):
    #print(len(values))
    try:
        print(n)    
        ic = ee.ImageCollection('COPERNICUS/S2_SR').filterDate('2019-01-01', '2019-01-30').filterBounds(boundary).filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',20)).map(maskS2clouds);
        #maybe, do here 19
        #2019-01-01', '2019-01-30
        
        #ic_ndvi = ic.normalizedDifference(NDVI_bands).rename(['NDVI'])
        # ic_ndwi = ic.normalizedDifference(NDWI_bands).rename(['NDWI'])
        # ic_ndbi = ic.normalizedDifference(NDBI_bands).rename(['NDBI'])
        # ic_all_bands = ic.select(['B2','B3','B4','B8','B11'],["blue","green","red","vnir","swir"])
        # #ic_ndvi_all_bands= ic_all_bands.addBands(ic_ndvi)
        # ic = ic_all_bands.addBands(ic_ndvi)
        #ic = ic.select(['B4','B3','B2'],["red","green","blue"])
        #ic = ic.select(['B4','B3','B2', 'B8','B11' ],["red","green","blue","vnir","swir"])
        ic = ic.select(['B2','B3','B4','B8','B11'],["blue","green","red","vnir","swir"])
        #ic = ic.filterBounds(geom) #sort("CLOUD_COVER")
        
        points = ee.FeatureCollection.randomPoints(geom, 100, n)

        img = ee.Image(ic.mosaic())
        img = ee.Image(ic.median())
        
        nir = img.select('vnir');
        red = img.select('red');
        green = img.select('green');       
        ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI');
        #ndwi = green.subtract(nir).divide(green.add(nir)).rename('NDWI')
        img = img.addBands(ndvi);
        #img = img.addBands(ndwi);
        
        #img = img.addBands(patch.divide(100))
        
        #img = img.addBands(patch)
        img = img.addBands(new_patch)


        # Import SRTM elevation data
        #elevation = ee.Image("USGS/SRTMGL1_003");

        # Calculate slope, aspect, and hillshade
        #topo = ee.Algorithms.Terrain(elevation);
        #img = img.addBands(topo).unmask(0);

        #print(name.getInfo())
        #print(points.size().getInfo())

        neighborhood = img.neighborhoodToArray(kernel);

        trainingSet= neighborhood.sampleRegions(collection=points, scale=10, tileScale=16)
        #featureNames = ["red","green","blue","class"]
        
        featureNames = ["blue","green","red","vnir","swir","NDVI","class"]
        
        
        if values[n] <= 0.1:
                trainFilePrefix = folder+'/validation/' + "patchh"  + '_' + str(n).zfill(4)

        if values[n] > 0.1 and values[n] < 0.3:
                trainFilePrefix = folder+'/testing/' + "patchh"  + '_' + str(n).zfill(4)

        if values[n] >= 0.3:
                trainFilePrefix = folder+'/training/' + "patchh"  + '_' + str(n).zfill(4)


        
        #print(featureNames.getInfo())
        #print(trainingSet.first().getInfo())
        #print("-----------------------")
        #print(trainingSet.toList(500).get(1).getInfo())

        #trainFilePrefix = folder+'/training/' + "apatch"  + '_' + str(n).zfill(4)


        trainingTask = ee.batch.Export.table.toCloudStorage(collection=trainingSet,
                                                        description="ppatch"+str(n),
                                                        fileNamePrefix=trainFilePrefix,
                                                        bucket=outputBucket,
                                                        fileFormat='TFRecord',
                                                        selectors=featureNames)
        
        trainingTask.start()
    except:
        pass

print(ee.batch.Task.list())
