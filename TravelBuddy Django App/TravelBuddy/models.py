from django.db import models


class City(models.Model):
    city = models.CharField(max_length=500)
    bustours = models.FloatField()
    naturewildlifetours = models.FloatField()
    biketours = models.FloatField()
    ecotours = models.FloatField()
    adrenalineextremetours = models.FloatField()
    gearrentals = models.FloatField()
    hikingcampingtours = models.FloatField()
    wdatvoffroadtours = models.FloatField()
    bikingtrails = models.FloatField()
    boattours = models.FloatField()
    golfcourses = models.FloatField()
    climbingtours = models.FloatField()
    fishingcharterstours = models.FloatField()
    beaches = models.FloatField()
    kayakingcanoeing = models.FloatField()
    standuppaddleboarding = models.FloatField()
    airtours = models.FloatField()
    boatrentals = models.FloatField()
    canyoningrappellingtours = models.FloatField()
    dolphinwhalewatching = models.FloatField()
    gondolacruises = models.FloatField()
    hikingtrails = models.FloatField()
    horsebackridingtours = models.FloatField()
    offroadatvtrails = models.FloatField()
    skisnowtours = models.FloatField()
    sportscampsclinics = models.FloatField()
    waterskiingjetskiing = models.FloatField()
    ziplineaerialadventureparks = models.FloatField()
    zoos = models.FloatField()
    theaters = models.FloatField()
    performances = models.FloatField()
    bluesclubsbars = models.FloatField()
    symphonies = models.FloatField()
    winetourstastings = models.FloatField()
    foodtours = models.FloatField()
    beertastingstours = models.FloatField()
    wineriesvineyards = models.FloatField()
    distillerytours = models.FloatField()
    breweries = models.FloatField()
    cookingclasses = models.FloatField()
    distilleries = models.FloatField()
    farmersmarkets = models.FloatField()
    otherfooddrink = models.FloatField()
    winebars = models.FloatField()
    culturalevents = models.FloatField()
    fooddrinkfestivals = models.FloatField()
    lessonsworkshops = models.FloatField()
    paintpotterystudios = models.FloatField()
    giftspecialtyshops = models.FloatField()
    shoppingmalls = models.FloatField()
    artgalleries = models.FloatField()
    fleastreetmarkets = models.FloatField()
    antiquestores = models.FloatField()
    departmentstores = models.FloatField()
    taxisshuttles = models.FloatField()
    bustransportation = models.FloatField()
    railways = models.FloatField()
    masstransportationsystems = models.FloatField()
    tramways = models.FloatField()
    libraries = models.FloatField()
    conferenceconventioncenters = models.FloatField()
    visitorcenters = models.FloatField()
    airportlounges = models.FloatField()
    daytrips = models.FloatField()
    privatetours = models.FloatField()
    watersports = models.FloatField()
    runningtours = models.FloatField()
    otheroutdooractivities = models.FloatField()
    balloonrides = models.FloatField()
    scubasnorkeling = models.FloatField()
    speedboatstours = models.FloatField()
    surfingwindsurfingkitesurfing = models.FloatField()
    horsedrawncarriagetours = models.FloatField()
    safaris = models.FloatField()
    concerts = models.FloatField()
    comedyclubs = models.FloatField()
    jazzclubsbars = models.FloatField()
    operas = models.FloatField()
    cabarets = models.FloatField()
    dinnertheaters = models.FloatField()
    ballets = models.FloatField()
    countrywesternbars = models.FloatField()
    pianobars = models.FloatField()
    coffeeteatours = models.FloatField()
    musicfestivals = models.FloatField()
    sportingevents = models.FloatField()
    shoppingtours = models.FloatField()
    factoryoutlets = models.FloatField()
    fashionshowstours = models.FloatField()
    ferries = models.FloatField()
    culturaltours = models.FloatField()
    historicalheritagetours = models.FloatField()
    walkingtours = models.FloatField()
    riverraftingtubing = models.FloatField()
    beachpoolclubs = models.FloatField()
    parasailingparagliding = models.FloatField()
    swimwithdolphins = models.FloatField()
    equestriantrails = models.FloatField()
    joggingpathstracks = models.FloatField()
    motorcycletrails = models.FloatField()
    scenicdrives = models.FloatField()
    exhibitions = models.FloatField()
    portsofcalltours = models.FloatField()
    ducktours = models.FloatField()
    airportshops = models.FloatField()
    submarinetours = models.FloatField()
    sharkdiving = models.FloatField()
    skisnowboardareas = models.FloatField()
    citytours = models.FloatField()
    archaeologytours = models.FloatField()
    multidaytours = models.FloatField()
    cirquedusoleilshows = models.FloatField()
    selfguidedtoursrentals = models.FloatField()
    luaus = models.FloatField()
    seasonalfireworks = models.FloatField()
    crosscountryskiareas = models.FloatField()
    ghostvampiretours = models.FloatField()
    ofpublicgreenspaceparksandgardens = models.FloatField()
    oftotalnationalcountrypopulationlivinginthecity = models.FloatField()
    averagedailynumberofvisitstotop5artexhibitions = models.FloatField()
    averageincomepercapitaperyearppp = models.FloatField()
    bookshops = models.FloatField()
    cinemascreens = models.FloatField()
    cinemas = models.FloatField()
    communitycentres = models.FloatField()
    creativeindustriesemployment = models.FloatField()
    educationlevelwithdegreelevelorhigher = models.FloatField()
    estimatedattendanceatmaincarnivalfestival = models.FloatField()
    estimatedattendanceatmaincarnivalfestivalasofcitypopulation = models.FloatField()
    festivalsandcelebrations = models.FloatField()
    filmfestivals = models.FloatField()
    foreignbornpopulation = models.FloatField()
    gdppppmillion = models.FloatField()
    geographicalareasizesqkm = models.FloatField()
    majorconcerthalls = models.FloatField()
    museumsgalleriesattendanceworkingagepopulationattendingatleastonceayear = models.FloatField()
    nationalmuseums = models.FloatField()
    nightclubsdiscosanddancehalls = models.FloatField()
    numberofadmissionsatalltheatresperyear = models.FloatField()
    numberofadmissionsatmainfilmfestival = models.FloatField()
    numberofartiststudiocomplexes = models.FloatField()
    numberofbars = models.FloatField()
    numberofbarsper100000population = models.FloatField()
    numberofbookloansbypubliclibrariesperyearmillion = models.FloatField()
    numberofbooktitlespublishedinthecountryinayear = models.FloatField()
    numberofbookshopsper100000population = models.FloatField()
    numberofcinemaadmissionsperyear = models.FloatField()
    numberofcinemascreensper100000population = models.FloatField()
    numberofdanceperformancesperyear = models.FloatField()
    numberoffilmsgiventheatricalreleaseinthecountryinayear = models.FloatField()
    numberofforeignfilmsgiventheatricalreleaseinthecountryinayear = models.FloatField()
    numberofhouseholds = models.FloatField()
    numberofinternationalstudentsstudyinginthecity = models.FloatField()
    numberofinternationaltouristsperyear = models.FloatField()
    numberofinternationaltouristsperyearasofcitypopulation = models.FloatField()
    numberofmarkets = models.FloatField()
    numberofmichelinstarrestaurants = models.FloatField()
    numberofmusicperformancesperyear = models.FloatField()
    numberofnonprofessionaldanceschools = models.FloatField()
    numberofotherheritagehistoricalsites = models.FloatField()
    numberofperformingartsdancerehearsalspaces = models.FloatField()
    numberofpubliclibrariesper100000population = models.FloatField()
    numberofrestaurants = models.FloatField()
    numberofrestaurantsper100000population = models.FloatField()
    numberofstudentsenrolledinspecialistartdesignpublicinstitutions = models.FloatField()
    numberofstudentsofartdesigndegreecoursesatgeneralistuniversities = models.FloatField()
    numberoftheatricalperformancesatalltheatresperyear = models.FloatField()
    othermuseums = models.FloatField()
    publiclibraries = models.FloatField()
    specialistprivateculturalheestablishments = models.FloatField()
    specialistpublicculturalheestablishments = models.FloatField()
    totalnumberofmuseums = models.FloatField()
    unescoworldheritagesites = models.FloatField()
    cityareakm2 = models.FloatField()
    metroareakm2 = models.FloatField()
    foreignborn = models.FloatField()
    gdppercapitathousandspppratesperresident = models.FloatField()
    unemploymentrate = models.FloatField()
    povertyrate = models.FloatField()
    masstransitcommuters = models.FloatField()
    majorairports = models.FloatField()
    majorports = models.FloatField()
    percentofpopulationwithhighereducation = models.FloatField()
    highereducationinstitutions = models.FloatField()
    totaltouristsannuallymillions = models.FloatField()
    foreigntouristsannuallymillions = models.FloatField()
    domestictouristsannuallymillions = models.FloatField()
    annualtourismrevenueusbillions = models.FloatField()
    hotelroomsthousands = models.FloatField()
    infantmortalitydeathsper1000births = models.FloatField()
    lifeexpectancyinyearsmale = models.FloatField()
    lifeexpectancyinyearsfemale = models.FloatField()
    physiciansper100000people = models.FloatField()
    numberofhospitals = models.FloatField()
    antismokinglegislation = models.FloatField()
    numberofmuseums = models.FloatField()
    numberofculturalandartsorganizations = models.FloatField()
    greenspaceskm2 = models.FloatField()
    airquality = models.FloatField()
    lawsorregulationstoimproveenergyefficiency = models.FloatField()
    retrofittedcityvehiclefleet = models.FloatField()
    bikeshareprogram = models.FloatField()
    bike = models.FloatField()
    bikeshare = models.FloatField()
    boat = models.FloatField()
    bus = models.FloatField()
    buses = models.FloatField()
    commuterrail = models.FloatField()
    cycling = models.FloatField()
    ferry = models.FloatField()
    funiculars = models.FloatField()
    handicapshuttle = models.FloatField()
    intercitycarship = models.FloatField()
    jeepney = models.FloatField()
    lightrail = models.FloatField()
    metro = models.FloatField()
    minibus = models.FloatField()
    monorail = models.FloatField()
    pedicab = models.FloatField()
    rail = models.FloatField()
    railway = models.FloatField()
    regionalrail = models.FloatField()
    rickshaw = models.FloatField()
    rickshawvan = models.FloatField()
    skytrain = models.FloatField()
    streetcar = models.FloatField()
    streetcars = models.FloatField()
    subway = models.FloatField()
    taxi = models.FloatField()
    train = models.FloatField()
    tram = models.FloatField()
    tramway = models.FloatField()
    trolley = models.FloatField()
    trolleybus = models.FloatField()
    metrorail = models.FloatField()
    subways = models.FloatField()


    def __str__(self):
        return self.city

class Attraction(models.Model):
    city = models.CharField(max_length=500)
    bustours = models.CharField(max_length=500)
    naturewildlifetours = models.CharField(max_length=500)
    biketours = models.CharField(max_length=500)
    ecotours = models.CharField(max_length=500)
    adrenalineextremetours = models.CharField(max_length=500)
    gearrentals = models.CharField(max_length=500)
    hikingcampingtours = models.CharField(max_length=500)
    wdatvoffroadtours = models.CharField(max_length=500)
    bikingtrails = models.CharField(max_length=500)
    boattours = models.CharField(max_length=500)
    golfcourses = models.CharField(max_length=500)
    climbingtours = models.CharField(max_length=500)
    fishingcharterstours = models.CharField(max_length=500)
    beaches = models.CharField(max_length=500)
    kayakingcanoeing = models.CharField(max_length=500)
    standuppaddleboarding = models.CharField(max_length=500)
    airtours = models.CharField(max_length=500)
    boatrentals = models.CharField(max_length=500)
    canyoningrappellingtours = models.CharField(max_length=500)
    dolphinwhalewatching = models.CharField(max_length=500)
    gondolacruises = models.CharField(max_length=500)
    hikingtrails = models.CharField(max_length=500)
    horsebackridingtours = models.CharField(max_length=500)
    offroadatvtrails = models.CharField(max_length=500)
    skisnowtours = models.CharField(max_length=500)
    sportscampsclinics = models.CharField(max_length=500)
    waterskiingjetskiing = models.CharField(max_length=500)
    ziplineaerialadventureparks = models.CharField(max_length=500)
    zoos = models.CharField(max_length=500)
    theaters = models.CharField(max_length=500)
    performances = models.CharField(max_length=500)
    bluesclubsbars = models.CharField(max_length=500)
    symphonies = models.CharField(max_length=500)
    winetourstastings = models.CharField(max_length=500)
    foodtours = models.CharField(max_length=500)
    beertastingstours = models.CharField(max_length=500)
    wineriesvineyards = models.CharField(max_length=500)
    distillerytours = models.CharField(max_length=500)
    breweries = models.CharField(max_length=500)
    cookingclasses = models.CharField(max_length=500)
    distilleries = models.CharField(max_length=500)
    farmersmarkets = models.CharField(max_length=500)
    otherfooddrink = models.CharField(max_length=500)
    winebars = models.CharField(max_length=500)
    culturalevents = models.CharField(max_length=500)
    fooddrinkfestivals = models.CharField(max_length=500)
    lessonsworkshops = models.CharField(max_length=500)
    paintpotterystudios = models.CharField(max_length=500)
    giftspecialtyshops = models.CharField(max_length=500)
    shoppingmalls = models.CharField(max_length=500)
    artgalleries = models.CharField(max_length=500)
    fleastreetmarkets = models.CharField(max_length=500)
    antiquestores = models.CharField(max_length=500)
    departmentstores = models.CharField(max_length=500)
    taxisshuttles = models.CharField(max_length=500)
    bustransportation = models.CharField(max_length=500)
    railways = models.CharField(max_length=500)
    masstransportationsystems = models.CharField(max_length=500)
    tramways = models.CharField(max_length=500)
    libraries = models.CharField(max_length=500)
    conferenceconventioncenters = models.CharField(max_length=500)
    visitorcenters = models.CharField(max_length=500)
    airportlounges = models.CharField(max_length=500)
    daytrips = models.CharField(max_length=500)
    privatetours = models.CharField(max_length=500)
    watersports = models.CharField(max_length=500)
    runningtours = models.CharField(max_length=500)
    otheroutdooractivities = models.CharField(max_length=500)
    balloonrides = models.CharField(max_length=500)
    scubasnorkeling = models.CharField(max_length=500)
    speedboatstours = models.CharField(max_length=500)
    surfingwindsurfingkitesurfing = models.CharField(max_length=500)
    horsedrawncarriagetours = models.CharField(max_length=500)
    safaris = models.CharField(max_length=500)
    concerts = models.CharField(max_length=500)
    comedyclubs = models.CharField(max_length=500)
    jazzclubsbars = models.CharField(max_length=500)
    operas = models.CharField(max_length=500)
    cabarets = models.CharField(max_length=500)
    dinnertheaters = models.CharField(max_length=500)
    ballets = models.CharField(max_length=500)
    countrywesternbars = models.CharField(max_length=500)
    pianobars = models.CharField(max_length=500)
    coffeeteatours = models.CharField(max_length=500)
    musicfestivals = models.CharField(max_length=500)
    sportingevents = models.CharField(max_length=500)
    shoppingtours = models.CharField(max_length=500)
    factoryoutlets = models.CharField(max_length=500)
    fashionshowstours = models.CharField(max_length=500)
    ferries = models.CharField(max_length=500)
    culturaltours = models.CharField(max_length=500)
    historicalheritagetours = models.CharField(max_length=500)
    walkingtours = models.CharField(max_length=500)
    riverraftingtubing = models.CharField(max_length=500)
    beachpoolclubs = models.CharField(max_length=500)
    parasailingparagliding = models.CharField(max_length=500)
    swimwithdolphins = models.CharField(max_length=500)
    equestriantrails = models.CharField(max_length=500)
    joggingpathstracks = models.CharField(max_length=500)
    motorcycletrails = models.CharField(max_length=500)
    scenicdrives = models.CharField(max_length=500)
    exhibitions = models.CharField(max_length=500)
    portsofcalltours = models.CharField(max_length=500)
    ducktours = models.CharField(max_length=500)
    airportshops = models.CharField(max_length=500)
    submarinetours = models.CharField(max_length=500)
    sharkdiving = models.CharField(max_length=500)
    skisnowboardareas = models.CharField(max_length=500)
    citytours = models.CharField(max_length=500)
    archaeologytours = models.CharField(max_length=500)
    multidaytours = models.CharField(max_length=500)
    cirquedusoleilshows = models.CharField(max_length=500)
    selfguidedtoursrentals = models.CharField(max_length=500)
    luaus = models.CharField(max_length=500)
    seasonalfireworks = models.CharField(max_length=500)
    crosscountryskiareas = models.CharField(max_length=500)
    ghostvampiretours = models.CharField(max_length=500)

    def __str__(self):
        return self.city


class Restaurant(models.Model):
    anger = models.FloatField()
    anticipation = models.FloatField()
    city = models.FloatField()
    disgust = models.FloatField()
    fear = models.FloatField()
    joy = models.FloatField()
    latitude = models.FloatField()
    longitude = models.FloatField()
    name = models.CharField(max_length=500)
    negative = models.FloatField()
    positive = models.FloatField()
    rating = models.FloatField()
    sadness = models.FloatField()
    surprise = models.FloatField()
    trust = models.FloatField()
    abruzzese = models.FloatField()
    acaibowls = models.FloatField()
    african = models.FloatField()
    andalusian = models.FloatField()
    apulian = models.FloatField()
    arabian = models.FloatField()
    argentine = models.FloatField()
    armenian = models.FloatField()
    arroceria_paella = models.FloatField()
    asianfusion = models.FloatField()
    asturian = models.FloatField()
    australian = models.FloatField()
    austrian = models.FloatField()
    bagels = models.FloatField()
    bakeries = models.FloatField()
    bars = models.FloatField()
    basque = models.FloatField()
    bavarian = models.FloatField()
    bbq = models.FloatField()
    beachbars = models.FloatField()
    bedbreakfast = models.FloatField()
    beer_and_wine = models.FloatField()
    beerbar = models.FloatField()
    beergarden = models.FloatField()
    beergardens = models.FloatField()
    beerhall = models.FloatField()
    beisl = models.FloatField()
    belgian = models.FloatField()
    bento = models.FloatField()
    bistros = models.FloatField()
    bookstores = models.FloatField()
    brasseries = models.FloatField()
    brazilian = models.FloatField()
    breakfast_brunch = models.FloatField()
    breweries = models.FloatField()
    brewpubs = models.FloatField()
    british = models.FloatField()
    bubbletea = models.FloatField()
    buffets = models.FloatField()
    burgers = models.FloatField()
    burmese = models.FloatField()
    butcher = models.FloatField()
    cafes = models.FloatField()
    cafeteria = models.FloatField()
    cajun = models.FloatField()
    cakeshop = models.FloatField()
    cambodian = models.FloatField()
    canteen = models.FloatField()
    cantonese = models.FloatField()
    caribbean = models.FloatField()
    catalan = models.FloatField()
    catering = models.FloatField()
    centralbrazilian = models.FloatField()
    champagne_bars = models.FloatField()
    cheese = models.FloatField()
    cheesesteaks = models.FloatField()
    chicken_wings = models.FloatField()
    chickenshop = models.FloatField()
    chilean = models.FloatField()
    chimneycakes = models.FloatField()
    chinese = models.FloatField()
    chocolate = models.FloatField()
    churros = models.FloatField()
    cideries = models.FloatField()
    cocktailbars = models.FloatField()
    coffee = models.FloatField()
    coffeeroasteries = models.FloatField()
    colombian = models.FloatField()
    comfortfood = models.FloatField()
    congee = models.FloatField()
    cookingschools = models.FloatField()
    corsican = models.FloatField()
    creperies = models.FloatField()
    cuban = models.FloatField()
    cucinacampana = models.FloatField()
    culturalcenter = models.FloatField()
    cupcakes = models.FloatField()
    currysausage = models.FloatField()
    czech = models.FloatField()
    danceclubs = models.FloatField()
    danish = models.FloatField()
    delicatessen = models.FloatField()
    delis = models.FloatField()
    desserts = models.FloatField()
    dimsum = models.FloatField()
    diners = models.FloatField()
    distilleries = models.FloatField()
    divebars = models.FloatField()
    dominican = models.FloatField()
    donburi = models.FloatField()
    donuts = models.FloatField()
    dumplings = models.FloatField()
    eastern_european = models.FloatField()
    easterngerman = models.FloatField()
    egyptian = models.FloatField()
    emilian = models.FloatField()
    empanadas = models.FloatField()
    eritrean = models.FloatField()
    ethiopian = models.FloatField()
    eventservices = models.FloatField()
    falafel = models.FloatField()
    farmersmarket = models.FloatField()
    filipino = models.FloatField()
    fishnchips = models.FloatField()
    flatbread = models.FloatField()
    florists = models.FloatField()
    fondue = models.FloatField()
    food = models.FloatField()
    food_court = models.FloatField()
    foodstands = models.FloatField()
    foodtrucks = models.FloatField()
    french = models.FloatField()
    friterie = models.FloatField()
    galician = models.FloatField()
    galleries = models.FloatField()
    gastropubs = models.FloatField()
    gelato = models.FloatField()
    georgian = models.FloatField()
    german = models.FloatField()
    giblets = models.FloatField()
    gluten_free = models.FloatField()
    gokarts = models.FloatField()
    gourmet = models.FloatField()
    greek = models.FloatField()
    grocery = models.FloatField()
    guamanian = models.FloatField()
    hainan = models.FloatField()
    hakka = models.FloatField()
    halal = models.FloatField()
    hawaiian = models.FloatField()
    hessian = models.FloatField()
    heuriger = models.FloatField()
    himalayan = models.FloatField()
    hkcafe = models.FloatField()
    hookah_bars = models.FloatField()
    horumon = models.FloatField()
    hotdog = models.FloatField()
    hotdogs = models.FloatField()
    hotels = models.FloatField()
    hotelstravel = models.FloatField()
    hotpot = models.FloatField()
    hungarian = models.FloatField()
    iberian = models.FloatField()
    icecream = models.FloatField()
    importedfood = models.FloatField()
    indonesian = models.FloatField()
    indpak = models.FloatField()
    international = models.FloatField()
    intlgrocery = models.FloatField()
    irish = models.FloatField()
    irish_pubs = models.FloatField()
    italian = models.FloatField()
    izakaya = models.FloatField()
    japacurry = models.FloatField()
    japanese = models.FloatField()
    jazzandblues = models.FloatField()
    jewish = models.FloatField()
    juicebars = models.FloatField()
    karaoke = models.FloatField()
    kebab = models.FloatField()
    kopitiam = models.FloatField()
    korean = models.FloatField()
    kosher = models.FloatField()
    lahmacun = models.FloatField()
    landmarks = models.FloatField()
    laotian = models.FloatField()
    latin = models.FloatField()
    lebanese = models.FloatField()
    localflavor = models.FloatField()
    lounges = models.FloatField()
    lumbard = models.FloatField()
    lyonnais = models.FloatField()
    malaysian = models.FloatField()
    markets = models.FloatField()
    meatballs = models.FloatField()
    meats = models.FloatField()
    mediterranean = models.FloatField()
    mexican = models.FloatField()
    mideastern = models.FloatField()
    milkbars = models.FloatField()
    modern_australian = models.FloatField()
    modern_european = models.FloatField()
    mongolian = models.FloatField()
    moroccan = models.FloatField()
    museums = models.FloatField()
    musicvenues = models.FloatField()
    napoletana = models.FloatField()
    nasilemak = models.FloatField()
    newamerican = models.FloatField()
    newcanadian = models.FloatField()
    newmexican = models.FloatField()
    newzealand = models.FloatField()
    nightfood = models.FloatField()
    noodles = models.FloatField()
    northernbrazilian = models.FloatField()
    northerngerman = models.FloatField()
    northernmexican = models.FloatField()
    norwegian = models.FloatField()
    nyonya = models.FloatField()
    okonomiyaki = models.FloatField()
    onigiri = models.FloatField()
    opensandwiches = models.FloatField()
    oriental = models.FloatField()
    ottomancuisine = models.FloatField()
    pakistani = models.FloatField()
    panasian = models.FloatField()
    pancakes = models.FloatField()
    pastashops = models.FloatField()
    persian = models.FloatField()
    peruvian = models.FloatField()
    piemonte = models.FloatField()
    pierogis = models.FloatField()
    pita = models.FloatField()
    pizza = models.FloatField()
    poke = models.FloatField()
    polish = models.FloatField()
    popuprestaurants = models.FloatField()
    portuguese = models.FloatField()
    poutineries = models.FloatField()
    provencal = models.FloatField()
    pubfood = models.FloatField()
    publicmarkets = models.FloatField()
    pubs = models.FloatField()
    puertorican = models.FloatField()
    ramen = models.FloatField()
    raw_food = models.FloatField()
    restaurants = models.FloatField()
    rhinelandian = models.FloatField()
    rodizios = models.FloatField()
    roman = models.FloatField()
    rotisserie_chicken = models.FloatField()
    russian = models.FloatField()
    salad = models.FloatField()
    salumerie = models.FloatField()
    salvadoran = models.FloatField()
    sandwiches = models.FloatField()
    sardinian = models.FloatField()
    saunas = models.FloatField()
    scandinavian = models.FloatField()
    schnitzel = models.FloatField()
    scottish = models.FloatField()
    seafood = models.FloatField()
    seafoodmarkets = models.FloatField()
    shanghainese = models.FloatField()
    shoppingcenters = models.FloatField()
    sicilian = models.FloatField()
    signature_cuisine = models.FloatField()
    singaporean = models.FloatField()
    slovakian = models.FloatField()
    smokehouse = models.FloatField()
    soba = models.FloatField()
    social_clubs = models.FloatField()
    soulfood = models.FloatField()
    soup = models.FloatField()
    southafrican = models.FloatField()
    southern = models.FloatField()
    souvenirs = models.FloatField()
    spanish = models.FloatField()
    spas = models.FloatField()
    sports_clubs = models.FloatField()
    sportsbars = models.FloatField()
    steak = models.FloatField()
    streetvendors = models.FloatField()
    sud_ouest = models.FloatField()
    sushi = models.FloatField()
    swabian = models.FloatField()
    swedish = models.FloatField()
    swimmingpools = models.FloatField()
    swissfood = models.FloatField()
    syrian = models.FloatField()
    szechuan = models.FloatField()
    tabernas = models.FloatField()
    tabletopgames = models.FloatField()
    tacos = models.FloatField()
    taiwanese = models.FloatField()
    tapas = models.FloatField()
    tapasmallplates = models.FloatField()
    tastingclasses = models.FloatField()
    tea = models.FloatField()
    tempura = models.FloatField()
    teochew = models.FloatField()
    teppanyaki = models.FloatField()
    texmex = models.FloatField()
    thai = models.FloatField()
    themedcafes = models.FloatField()
    tikibars = models.FloatField()
    tonkatsu = models.FloatField()
    toys = models.FloatField()
    tradamerican = models.FloatField()
    traditional_swedish = models.FloatField()
    trattorie = models.FloatField()
    trinidadian = models.FloatField()
    turkish = models.FloatField()
    tuscan = models.FloatField()
    udon = models.FloatField()
    unagi = models.FloatField()
    vegan = models.FloatField()
    vegetarian = models.FloatField()
    venezuelan = models.FloatField()
    venues = models.FloatField()
    vermouthbars = models.FloatField()
    vietnamese = models.FloatField()
    vintage = models.FloatField()
    waffles = models.FloatField()
    westernjapanese = models.FloatField()
    whiskeybars = models.FloatField()
    wine_bars = models.FloatField()
    wineries = models.FloatField()
    wok = models.FloatField()
    wraps = models.FloatField()
    yakiniku = models.FloatField()
    yakitori = models.FloatField()

    def __str__(self):
        return self.name