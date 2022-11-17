// Aggregate GRIDMET climate data to HUC8 level

//////////////////////// Import data ////////////////////////

// Read in shapefiles
var huc8_outline = ee.FeatureCollection("users/mdgaines/outline_clean_paper2");
var huc8 = ee.FeatureCollection("users/mdgaines/HUC08_paper2");

// Import GRIDMET data
var GRIDMET_data = ee.ImageCollection('IDAHO_EPSCOR/GRIDMET')
                  .filterBounds(huc8_outline)
                  .filter(ee.Filter.date('1979-01-01', '2022-09-01'))
                  .map(function(img) {return img.clip(huc8_outline)});
                  

//////////////////////// Average and download data ////////////////////////

// general function for calculating seasonal average climate variable
var winterAvgFunc = function(yr, yr_str, s_month, e_month, szn, gmet_var, var_name){

  var next_yr = (yr + 1).toString();
  var sznAvg = GRIDMET_data.select(gmet_var)
                          .filter(ee.Filter.date(yr_str + '-12-01', next_yr + '-03-01'))
                          .reduce(ee.Reducer.mean());
  
  // Calculate huc avg
  var zonal_avg = sznAvg.reduceRegions({
    collection: huc8,
    reducer: ee.Reducer.mean(),
    scale: 4000 
  });
  
  var zonal_avg_out = zonal_avg.select(['.*'],null,false);
  // make year_szn string
  var yr_szn_str = yr_str + szn;
  // set file name
  var desc = [yr_szn_str, var_name, 'AVG'].join('_');

  zonal_avg_out = zonal_avg_out.map(function(feature){
    return feature.set('Yr_Szn',yr_szn_str);
  });
  
  // save to drive as csv
  Export.table.toDrive({
    collection: zonal_avg_out,
    description: desc,
    folder: 'GRIDMET_YR_AVG',
    fileFormat: 'CSV'
  });
  
};

// general function for calculating seasonal average climate variable
var sznAvgFunc = function(yr, yr_str, s_month, e_month, szn, gmet_var, var_name){

  var sznAvg = GRIDMET_data.select(gmet_var)
                        .filter(ee.Filter.calendarRange(s_month,e_month,'month'))
                        .filter(ee.Filter.calendarRange(yr,yr,'year'))
                        .reduce(ee.Reducer.mean());
                        
  print(sznAvg);
  
  // Calculate huc avg
  var zonal_avg = sznAvg.reduceRegions({
    collection: huc8,
    reducer: ee.Reducer.mean(),
    scale: 4000 
  });
  
  var zonal_avg_out = zonal_avg.select(['.*'],null,false);
  // make year_szn string
  var yr_szn_str = yr_str + szn;
  // set file name
  var desc = [yr_szn_str, var_name, 'AVG'].join('_');

  zonal_avg_out = zonal_avg_out.map(function(feature){
    return feature.set('Yr_Szn',yr_szn_str);
  });
  
  // save to drive as csv
  Export.table.toDrive({
    collection: zonal_avg_out,
    description: desc,
    folder: 'GRIDMET_YR_AVG',
    fileFormat: 'CSV'
  });
  
};

// general function for calculating seasonal precip total HUC average
var winterPrecipAvgFunc = function(yr, yr_str, s_month, e_month, szn, gmet_var, var_name){

  var next_yr = (yr + 1).toString();
  var sznSum = GRIDMET_data.select(gmet_var)
                          .filter(ee.Filter.date(yr_str + '-12-01', next_yr + '-03-01'))
                          .reduce(ee.Reducer.sum());
  
  // Calculate huc avg
  var zonal_avg = sznSum.reduceRegions({
    collection: huc8,
    reducer: ee.Reducer.mean(),
    scale: 4000 
  });
  
  var zonal_avg_out = zonal_avg.select(['.*'],null,false);
  // make year_szn string
  var yr_szn_str = yr_str + szn;
  // set file name
  var desc = [yr_szn_str, var_name, 'AVG'].join('_');

  zonal_avg_out = zonal_avg_out.map(function(feature){
    return feature.set('Yr_Szn',yr_szn_str);
  });
  
  // save to drive as csv
  Export.table.toDrive({
    collection: zonal_avg_out,
    description: desc,
    folder: 'GRIDMET_YR_AVG',
    fileFormat: 'CSV'
  });
  
};

// general function for calculating seasonal precip total HUC average
var sznPrecipAvgFunc = function(yr, yr_str, s_month, e_month, szn, gmet_var, var_name){

  var sznSum = GRIDMET_data.select(gmet_var)
                        .filter(ee.Filter.calendarRange(s_month,e_month,'month'))
                        .filter(ee.Filter.calendarRange(yr,yr,'year'))
                        .reduce(ee.Reducer.sum());
                        
  print(sznSum);
  
  // Calculate huc avg
  var zonal_avg = sznSum.reduceRegions({
    collection: huc8,
    reducer: ee.Reducer.mean(),
    scale: 4000 
  });
  
  var zonal_avg_out = zonal_avg.select(['.*'],null,false);
  // make year_szn string
  var yr_szn_str = yr_str + szn;
  // set file name
  var desc = [yr_szn_str, var_name, 'AVG'].join('_');

  zonal_avg_out = zonal_avg_out.map(function(feature){
    return feature.set('Yr_Szn',yr_szn_str);
  });
  
  // save to drive as csv
  Export.table.toDrive({
    collection: zonal_avg_out,
    description: desc,
    folder: 'GRIDMET_YR_AVG',
    fileFormat: 'CSV'
  });
  
};

//////////////////////// Loop through each year ////////////////////////

for(var yr=1985; yr<2023; yr++){
  // print(index)
  // var yr = years[index];
  var prev_yr = yr + 1;
  var yr_str = yr.toString();
  print(yr);
  
  // sznAvgFunc(yr,yr_str,3,5,'_Sp','tmmx','maxTemp');
  // sznAvgFunc(yr,yr_str,3,5,'_Sp','tmmn','minTemp');
  sznPrecipAvgFunc(yr,yr_str,3,5,'_Sp','pr','Pr');

  // SUMMER
  // sznAvgFunc(yr,yr_str,6,8,'_Su','tmmx','maxTemp');
  // sznAvgFunc(yr,yr_str,6,8,'_Su','tmmn','minTemp');
  sznPrecipAvgFunc(yr,yr_str,6,8,'_Su','pr','Pr');

  // FALL
  // sznAvgFunc(yr,yr_str,9,11,'_Fa','tmmx','maxTemp');
  // sznAvgFunc(yr,yr_str,9,11,'_Fa','tmmn','minTemp');
  sznPrecipAvgFunc(yr,yr_str,9,11,'_Fa','pr','Pr');

  // SPRING
  // winterAvgFunc(yr,yr_str,12,2,'_Wi','tmmx','maxTemp');
  // winterAvgFunc(yr,yr_str,12,2,'_Wi','tmmn','minTemp');
  winterPrecipAvgFunc(yr,yr_str,12,2,'_Wi','pr','Pr');
  
}


