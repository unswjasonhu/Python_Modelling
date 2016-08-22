var heatmap, image;


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
};

function getGridData() {
    var datetime = date.value + '_' + time.value;
    var url ='/modeling/get_estimates_data?input_datetime=' + datetime;
    var dataPoints = []
    var values = [];
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
            for (k = 1; k < array.length; k++) {
                var tempLat = array[k][0][0];
                //console.log(tempLat);
                var tempLng = array[k][0][1];
                //console.log(tempLng);
                var tempWeight = array[k][1][1];
                heatmappoint.push({location: new google.maps.LatLng(tempLat, tempLng), weight: tempWeight});
                values.push(tempWeight);
            }
            dataPoints = heatmappoint;
            //console.log(dataPoints);
        }
    });
    //console.log('Max is ', Math.max.apply(null, values), ', Min is ', Math.min.apply(null, values));
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
          colv1Array = data;
          $.each(data, function(index, t){
            for (k=0; k<t.length; k++) {
                colvlArray[k] = t[k];
            }
          });
        }
    }).responseText;

    var dataValues = colv1Array[date.value];
    
    var chartdata = new google.visualization.DataTable();
    chartdata.addColumn('string', 'Time');
    chartdata.addColumn('number', 'CO level');
    for (k=0; k < dataValues.length; k++) {
        var timestring = dataValues[k][0]+ ':00';
        chartdata.addRow([timestring,dataValues[k][1]]);
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
            position: 'in',
            alignment: 'end',
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
            },
        }
    };

  var chart = new google.visualization.LineChart(document.getElementById('curve_chart'));

  chart.draw(chartdata, options);
}
