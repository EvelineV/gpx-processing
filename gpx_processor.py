import logging
import pandas
import gpxpy
from tqdm import tqdm
from collections import defaultdict
from gpxpy.geo import distance as gpx_distance
import xml.etree.ElementTree as etree
import numpy as np


class GPXProcessor(object):
    """
        This class loads and parses .gpx files.
        Computes various quantities and returns
        the data as pandas DataFrame.

        BartP
    """
    earth_circum = 40075000  # m
    deg_to_m = earth_circum / 360

    def __init__(self, loglevel=logging.INFO):
        self.logger = logging.getLogger("GPXProcessor")
        self.logger.setLevel(loglevel)

    def gpx_to_dataframe(self, gpx_filename, hr_info=True, track_offset=0):
        with open(gpx_filename, 'r') as f:
            gpx_file = gpxpy.parse(f)
        
        data = defaultdict(list)
        columns = ['track', 'segment', 'point',
                   'latitude', 'longitude', 'elevation',
                   'time', 'speed', 'distance']
        
        #tracks, segments, points
        for track_id, track in enumerate(gpx_file.tracks):
            for segment_id, segment in enumerate(track.segments):
                for point_id, point in enumerate(segment.points):
                    data['track'].append(track_id + track_offset)
                    data['segment'].append(segment_id)
                    data['point'].append(point_id)
                    data['latitude'].append(point.latitude)
                    data['longitude'].append(point.longitude)
                    data['elevation'].append(point.elevation)
                    data['time'].append(point.time)
                    data['speed'].append(segment.get_speed(point_id))
                    
                    if point_id > 0:
                        d = gpx_distance(prev_lat,
                                         prev_lon,
                                         prev_ele,
                                         point.latitude,
                                         point.longitude,
                                         point.elevation)
                        data['distance'].append(d)
                    else:
                        data['distance'].append(0)
                        
                    prev_lat = point.latitude
                    prev_lon = point.longitude
                    prev_ele = point.elevation

        if hr_info:
            #gpxpy doesn't parse complex extensions.. ugly workaround
            root_tree = etree.parse(gpx_filename).getroot()
            for extension in root_tree.iter('{http://www.garmin.com/xmlschemas/TrackPointExtension/v1}hr'):
                data['heart_rate'].append(extension.text)
            
            assert len(data['heart_rate']) == len(data['point'])
            columns.append('heart_rate')
        
        return pandas.DataFrame(data, columns=columns)

    def add_additional_cols(self, df, lon_mean, lat_mean):
        def speed_to_pace(row):
            """m/s to min/km"""
            s = row['speed']
            if np.isnan(s) or s == 0:
                return np.nan
            return 1000/(s*60)

        df['pace'] = df.apply(speed_to_pace, axis=1)
        df['total_distance'] = np.cumsum(df['distance'])
     
        max_d_lon = (df['longitude'].max() - df['longitude'].min())
        max_d_lat = (df['latitude'].max() - df['latitude'].min())
        
        if max_d_lon > 0.25 or max_d_lat > 0.25:
            self.logger.warning("Lon/lat range larger than 0.2 deg, inaccurate conversion to km scale")

        df['x'] = (df['longitude'] - lon_mean) * self.deg_to_m * np.cos(lat_mean * np.pi / 180)
        df['y'] = (df['latitude'] - lat_mean) * self.deg_to_m
        
        return df

    def process_files(self, file_list, additional_info=True, hr_info=True):
        n_tracks_offset = 0
        for i, file in tqdm(enumerate(file_list)):
            df_temp = self.gpx_to_dataframe(file, hr_info=hr_info, track_offset= i + n_tracks_offset)
            n_tracks_offset += len(set(df_temp['track'])) - 1
            
            if i == 0:
                # set only once, need unique reference frame
                lon_mean, lat_mean = df_temp['longitude'].mean(), df_temp['latitude'].mean()

            if additional_info:
                df_temp = self.add_additional_cols(df_temp, lon_mean, lat_mean)
            
            if i == 0:
                df = df_temp
            else:
                df = df.append(df_temp)
                
        return df
