
var map, heatmap, marker, image;

var myLatLng = {lat: -33.92313, lng: 150.98812};
function initMap() {
  map = new google.maps.Map(document.getElementById('map'), {
    center: myLatLng,
    zoom: 11,
    scaleControl:true,
  });

  heatmap = new google.maps.visualization.HeatmapLayer({
      map: map,
      radius: 8,
      data: getGridData()
  });
  heatmap.setMap(map);

  home_image = '/images/markers/home.png';
  marker = new google.maps.Marker({
    position: myLatLng,
    map: map,
    icon: home_image,
    title: 'UNSW is my home'
  });
};

function getPlotUrl() {
    var datetime = date.value + ' ' + time.value;
    var url ='/modeling/generate_plot?input_datetime=' + datetime;
    var plotUrl = '';
    $.ajax({
        url: url,
        dataType: 'json',
        async: false,
        data: "",
        success: function(data) {
            plotUrl = data['success']
            console.log("Got plot url ", plotUrl, "for datetime ", datetime)
        }
    });
    return plotUrl;
}

function getGridData() {
    var datetime = date.value + '_' + time.value;
    var url ='/modeling/get_estimates_data?input_datetime=' + datetime;
    var dataPoints = []
    $.ajax({
        url: url,
        dataType: 'json',
        async: false,
        data: "",
        success: function(data) {
            var array = [];
            $.each(data, function(index, d) {
                array = d;
            });
            var heatmappoint = [];
            //tempLat = array[0][0][0];
            //tempLng = array[0][0][1];
            //tempWeight = array[0][1][1];
            //heatmappoint[0] = '{location: new google.maps.LatLng('+tempLat+','+tempLng+'), weight:'+tempWeight+'},';
            for (k = 1; k < array.length; k++) {
                var tempLat = array[k][0][0];
                //console.log(tempLat);
                var tempLng = array[k][0][1];
                //console.log(tempLng);
                var tempWeight = array[k][1][1];
                //console.log(tempWeight);
                //heatmappoint[0] += '{location: new google.maps.LatLng('+tempLat+','+tempLng+'), weight:'+tempWeight+'},';
                //console.log(heatmappoint[0]);
                heatmappoint.push({location: new google.maps.LatLng(tempLat, tempLng), weight: tempWeight});
            }
            //tempLat = array[array.length-1][0][0];
            //tempLng = array[array.length-1][0][1];
            //tempWeight = array[array.length-1][1][1];
            //heatmappoint[0] += '{location: new google.maps.LatLng('+tempLat+','+tempLng+'), weight:'+tempWeight+'}';
            //console.log(heatmappoint[0]);
            //heatmappoint
            dataPoints = heatmappoint;
            //console.log(dataPoints);
        }
    });
    //console.log(dataPoints);
    return dataPoints;
};


function drawChart() {
    var colvlArray = [];
    var charturl = '/modeling/get_estimates_data?input_date=' + date.value +'&lat=' +lat.value + '&lon=' + lng.value;
    console.log(charturl);
    $.ajax({
        url: charturl,
        dataType: "json",
        async: false,
        data: "",
        success: function(data) {

          $.each(data, function(index, t){
              for (k=0; k<t.length; k++) {
                  colvlArray[k] = t[k][1];
              }
              //console.log(colvlArray);
          });
        }

    }).responseText;
    var chartdata = new google.visualization.DataTable();
    chartdata.addColumn('string', 'Time');
    chartdata.addColumn('number', 'CO level');
    for (k=0; k<colvlArray.length; k++) {
        var timeString = (k+1) +':00';
        chartdata.addRow([timeString, colvlArray[k]]);
    }


    var options = {
        title: 'Carbon Monoxide Concentration over time at '+lat.value+','+lng.value,
        titleTextStyle: {
                bold: true,
                fontSize: 32,
        },
        backgroundColor: 'beige',
        lineWidth: 4,
        lineDashStyle: [5,2,3,2],
        legend: {
            position: 'bottom'
        },
        hAxis: {
            title: 'Time',
            textStyle: {
                bold: true,
                fontSize: 16,
                color: 'red',
            },
            titleTextStyle: {
                bold: true,
                fontSize: 20,
                color: '#1a237e',
            }
        },
        vAxis: {
            title: 'CO Level',
            textStyle: {
                bold: true,
                fontSize: 16,
                color: 'red',
            },
            titleTextStyle: {
                bold: true,
                fontSize: 20,
                color: '#1a237e',
            }
        }
    };

  var chart = new google.visualization.LineChart(document.getElementById('curve_chart'));

  chart.draw(chartdata, options);
}
