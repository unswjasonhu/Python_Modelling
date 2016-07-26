var map, heatmap, marker, image;


var myLatLng = {lat: -33.92313, lng: 150.98812};

//function initMap() {
//  map = new google.maps.Map(document.getElementById('map'), {
//    center: myLatLng,
//    zoom: 11,
//    scaleControl:true,
//  });
//
//  // Overlay parameters
//  var bounds = new google.maps.LatLngBounds(
//    new google.maps.LatLng(62.281819, -150.287132),
//    new google.maps.LatLng(62.400471, -150.005608));
//
//  var srcImage = getPlotUrl();
//  overlay = new EstimatesOverlay(bounds, srcImage, map);
//
//
//  //home marker params
//  home_image = '/images/markers/home.png';
//  marker = new google.maps.Marker({
//    position: myLatLng,
//    map: map,
//    icon: home_image,
//    title: 'UNSW is my home'
//  });
//};
//
//
///**
// * onAdd is called when the map's panes are ready and the overlay has been
// * added to the map.
// */
//EstimatesOverlay.prototype.onAdd = function() {
//
//  var div = document.createElement('div');
//  div.style.borderStyle = 'none';
//  div.style.borderWidth = '0px';
//  div.style.position = 'absolute';
//
//  // Create the img element and attach it to the div.
//  var img = document.createElement('img');
//  img.src = this.image_;
//  img.style.width = '100%';
//  img.style.height = '100%';
//  img.style.position = 'absolute';
//  div.appendChild(img);
//
//  this.div_ = div;
//
//  // Add the element to the "overlayLayer" pane.
//  var panes = this.getPanes();
//  panes.overlayLayer.appendChild(div);
//};
//
//EstimatesOverlay.prototype.draw = function() {
//
//  // We use the south-west and north-east
//  // coordinates of the overlay to peg it to the correct position and size.
//  // To do this, we need to retrieve the projection from the overlay.
//  var overlayProjection = this.getProjection();
//
//  // Retrieve the south-west and north-east coordinates of this overlay
//  // in LatLngs and convert them to pixel coordinates.
//  // We'll use these coordinates to resize the div.
//  var sw = overlayProjection.fromLatLngToDivPixel(this.bounds_.getSouthWest());
//  var ne = overlayProjection.fromLatLngToDivPixel(this.bounds_.getNorthEast());
//
//  // Resize the image's div to fit the indicated dimensions.
//  var div = this.div_;
//  div.style.left = sw.x + 'px';
//  div.style.top = ne.y + 'px';
//  div.style.width = (ne.x - sw.x) + 'px';
//  div.style.height = (sw.y - ne.y) + 'px';
//};
//
//// The onRemove() method will be called automatically from the API if
//// we ever set the overlay's map property to 'null'.
//EstimatesOverlay.prototype.onRemove = function() {
//  this.div_.parentNode.removeChild(this.div_);
//  this.div_ = null;
//};
//

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
            for (k = 1; k < array.length; k++) {
                var tempLat = array[k][0][0];
                //console.log(tempLat);
                var tempLng = array[k][0][1];
                //console.log(tempLng);
                var tempWeight = array[k][1][1];
                heatmappoint.push({location: new google.maps.LatLng(tempLat, tempLng), weight: tempWeight});
            }
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
