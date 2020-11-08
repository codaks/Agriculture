
moisture_of_soil = document.getElementById("moisture_of_soil")
temp_in_environment = document.getElementById("temp_in_environment")
PH_of_soil = document.getElementById("PH_of_soil")
temperature_of_soil = document.getElementById("temperature_of_soil")
humidity_in_environment = document.getElementById("humidity_in_environment")
light_intensity_in_environment = document.getElementById("light_intensity_in_environment")


userId = '1'
const database = firebase.database();
const usersRef = database.ref('/livedata');
usersRef.child(userId).set({
    moisture_of_soil: '14',
    temp_in_environment: '15',
    PH_of_soil: '16',
    temperature_of_soil: '17',
    humidity_in_environment: '18',
    light_intensity_in_environment: '19'
});



usersRef.on('child_changed', snapshot => {
    console.log('Child updated !');
    usersRef.orderByKey().limitToLast(1).on('value', snapshot => {
        data = snapshot.val()
        moisture_of_soil.innerHTML = data['1']['moisture_of_soil'];
        temp_in_environment.innerHTML = data['1']['temp_in_environment'];
        PH_of_soil.innerHTML = data['1']['PH_of_soil'];
        humidity_in_environment.innerHTML = data['1']['humidity_in_environment'];
        light_intensity_in_environment.innerHTML = data['1']['light_intensity_in_environment'];
        temperature_of_soil.innerHTML = data['1']['temperature_of_soil'];
    });
});